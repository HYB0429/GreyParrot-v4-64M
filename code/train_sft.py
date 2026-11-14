# -*- coding: utf-8 -*-
"""GreyParrot 64M SFT 训练器（MiniMind sft_t2t 格式，loss 只算助手回复）。

数据：prep_sft.py 产物 sft_{train,val}_{tokens,mask}.npy，定长 512，EOS 填充。
示例：
python train_sft.py --data-dir /workspace/data/sft_bin --init /workspace/runs/.../latest.pt \
  --out /workspace/runs/greyparrot64m_sft --lr 2e-5 --max-steps 8000 --compile --compile-target hidden-notsa
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from model_v6_fast import build_64m, VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="GreyParrot 64M SFT trainer")
    p.add_argument("--data-dir", required=True, help="含 sft_{train,val}_{tokens,mask}.npy 的目录")
    p.add_argument("--out", default="runs/greyparrot64m_sft")
    p.add_argument("--init", default="", help="预训练 checkpoint 路径，仅加载模型权重（优化器重新初始化）")
    p.add_argument("--resume", default="", help="SFT checkpoint 路径，完整恢复（含优化器/步数）")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=8000, help="optimizer steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ce-chunk", type=int, default=1024)
    p.add_argument("--rkm-scan-chunk", type=int, default=64)
    p.add_argument("--tsa-chunk", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compile-target", default="hidden-notsa",
                   choices=["hidden-notsa", "rkm", "ffn", "rkm-ffn"])
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-interval", type=int, default=250)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--final-save-seconds", type=float, default=0)
    p.add_argument("--final-save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-threads", type=int, default=8)
    return p.parse_args()


def save_checkpoint(path, model, optimizer, scheduler, step, best_val, cfg, args,
                    train_rng, val_rng, device):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), step=step, best_val=best_val,
                   cfg=cfg, args=vars(args), torch_version=torch.__version__,
                   rng_state=dict(train_rng=train_rng.bit_generator.state,
                                  val_rng=val_rng.bit_generator.state,
                                  torch=torch.get_rng_state()))
    if device.type == "cuda":
        payload["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.init and args.resume:
        raise ValueError("--init 与 --resume 只能二选一")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    use_bf16 = args.dtype == "bf16"
    if use_bf16 and device.type != "cuda":
        raise ValueError("bf16 模式仅用于 ROCm/CUDA")
    torch.set_num_threads(max(1, args.num_threads))
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.data_dir)
    tr_tok = np.load(d / "sft_train_tokens.npy", mmap_mode="r")
    tr_msk = np.load(d / "sft_train_mask.npy", mmap_mode="r")
    va_tok = np.load(d / "sft_val_tokens.npy", mmap_mode="r")
    va_msk = np.load(d / "sft_val_mask.npy", mmap_mode="r")
    T = tr_tok.shape[1]
    n_train = len(tr_tok)
    print(f"SFT train={n_train:,} samples, val={len(va_tok):,}, T={T}, device={device}, autocast={args.dtype}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    val_order = np.arange(len(va_tok))

    def make_batch(tok, msk, idx):
        t = torch.from_numpy(np.asarray(tok[idx], dtype=np.int64))
        m = torch.from_numpy(np.asarray(msk[idx], dtype=np.int64))
        y = torch.where(m == 1, t, torch.full_like(t, -100))
        x, y = t[:, :-1], y[:, 1:]
        if device.type == "cuda":
            x, y = x.pin_memory(), y.pin_memory()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def get_train_batch():
        idx = train_rng.integers(0, n_train, size=args.batch_size)
        return make_batch(tr_tok, tr_msk, idx)

    def get_val_batch(vi):
        idx = val_order[vi * args.batch_size:(vi + 1) * args.batch_size]
        return make_batch(va_tok, va_msk, idx)

    model, cfg, n_params = build_64m(verbose=True)
    cfg["rkm_scan_chunk"] = args.rkm_scan_chunk
    for blk in model.blocks:
        blk.dual.rkm.scan_chunk = args.rkm_scan_chunk
        blk.checkpoint_modules = set()
    model = model.to(device)
    tsa_chunk = args.tsa_chunk or None
    print(f"parameters={n_params:,}; TSA candidates/query={model.blocks[0].dual.tsa.pool_size}")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(dict(vocab=VOCAB, params=n_params, cfg=cfg, args=vars(args)),
                  f, ensure_ascii=False, indent=2)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith("theta") else decay).append(p)
    optim_groups = [dict(params=decay, weight_decay=args.weight_decay),
                    dict(params=no_decay, weight_decay=0.0)]
    fused_ok = device.type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95),
                                      eps=1e-8, fused=fused_ok)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
        fused_ok = False
    print("fused AdamW:", fused_ok)

    def lr_scale(step):
        if args.warmup and step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        progress = min(1.0, max(0.0, progress))
        floor = args.min_lr / args.lr
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    step, best_val = 0, float("inf")

    src = args.resume or args.init
    if src:
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        saved_cfg = ckpt.get("cfg", {})
        runtime_only = {"ce_chunk", "activation_checkpoint", "checkpoint_modules", "rkm_scan_chunk",
                        "tsa_vec_budget", "report_stats"}
        mismatched = {k: (v, cfg.get(k)) for k, v in saved_cfg.items()
                      if k not in runtime_only and k in cfg and cfg[k] != v}
        if mismatched:
            raise ValueError(f"模型配置不兼容: {mismatched}")
        model.load_state_dict(ckpt["model"])
        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            step, best_val = int(ckpt["step"]), float(ckpt.get("best_val", float("inf")))
            rng_state = ckpt.get("rng_state")
            if rng_state:
                train_rng.bit_generator.state = rng_state["train_rng"]
                torch.set_rng_state(rng_state["torch"])
                if device.type == "cuda" and "cuda" in rng_state:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
            print(f"resume: step={step}, best_val={best_val}（含优化器状态）")
        else:
            print(f"init: 从 {src} 仅加载模型权重（step={ckpt.get('step')}, "
                  f"best_val={ckpt.get('best_val')}），优化器全新初始化")

    if args.compile:
        target = args.compile_target
        if target == "hidden-notsa":
            for blk in model.blocks:
                if not getattr(blk.dual.tsa, "fused_attn_core", False):
                    blk.dual.tsa.attn_core = torch.compile(
                        blk.dual.tsa.attn_core, mode=args.compile_mode, dynamic=False)
                blk.dual.tsa.forward = torch.compiler.disable(blk.dual.tsa.forward)
            model.forward_hidden = torch.compile(model.forward_hidden,
                                                 mode=args.compile_mode, dynamic=False)
        else:
            targets = target.split("-")
            for blk in model.blocks:
                if "rkm" in targets:
                    blk.dual.rkm = torch.compile(blk.dual.rkm, mode=args.compile_mode, dynamic=False)
                if "tsa" in targets:
                    blk.dual.tsa = torch.compile(blk.dual.tsa, mode=args.compile_mode, dynamic=False)
                if "ffn" in targets:
                    blk.ffn = torch.compile(blk.ffn, mode=args.compile_mode, dynamic=False)
        print(f"torch.compile target={target}, mode={args.compile_mode}, dynamic=False")

    def autocast():
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=use_bf16)

    @torch.no_grad()
    def validate():
        model.eval()
        values = []
        for vi in range(min(args.val_batches, len(va_tok) // args.batch_size)):
            x, y = get_val_batch(vi)
            with autocast():
                loss, _ = model(x, y, chunk=tsa_chunk, ce_chunk=args.ce_chunk,
                                activation_checkpoint=False)
            values.append(loss.detach())
        model.train()
        return torch.stack(values).mean()

    log_path = out / "metrics.jsonl"
    last_time = time.perf_counter()
    train_start = last_time
    last_step = step
    running_loss = torch.zeros((), device=device)
    running_micro = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        if args.max_seconds > 0 and time.perf_counter() - train_start >= args.max_seconds:
            break
        for _ in range(args.grad_accum):
            x, y = get_train_batch()
            with autocast():
                loss, _ = model(x, y, chunk=tsa_chunk, ce_chunk=args.ce_chunk,
                                activation_checkpoint=False)
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            running_loss += loss.detach()
            running_micro += 1
        if args.grad_clip > 0:
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip,
                                               fused=(device.type == "cuda"))
            except (TypeError, RuntimeError):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        should_log = step % args.log_interval == 0 or step == 1
        should_val = args.val_interval > 0 and (step % args.val_interval == 0 or step == args.max_steps)
        if should_log or should_val:
            now = time.perf_counter()
            n_tokens = (step - last_step) * args.batch_size * T * args.grad_accum
            interval_s = max(1e-9, now - last_time)
            record = dict(step=step, train_loss=float(running_loss / max(1, running_micro)),
                          lr=optimizer.param_groups[0]["lr"], tokens_per_s=n_tokens / interval_s,
                          elapsed_s=now - last_time)
            if device.type == "cuda":
                record["train_peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
                torch.cuda.reset_peak_memory_stats(device)
            if should_val:
                val_loss = validate()
                record["val_loss"] = float(val_loss)
                if record["val_loss"] < best_val:
                    best_val = record["val_loss"]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            running_loss.zero_()
            running_micro = 0
            last_time, last_step = time.perf_counter(), step
        elapsed_total = time.perf_counter() - train_start
        final_phase = (args.max_seconds > 0 and args.final_save_seconds > 0 and
                       elapsed_total >= max(0.0, args.max_seconds - args.final_save_seconds))
        save_interval = args.final_save_interval if final_phase else args.save_interval
        if save_interval > 0 and (step % save_interval == 0 or step == args.max_steps):
            save_checkpoint(out / "latest.pt", model, optimizer, scheduler, step,
                            best_val, cfg, args, train_rng, val_rng, device)
            print("saved", out / "latest.pt", "step", step,
                  "final_phase", final_phase, flush=True)

    if args.max_steps == 0 or args.max_seconds > 0:
        save_checkpoint(out / "latest.pt", model, optimizer, scheduler, step,
                        best_val, cfg, args, train_rng, val_rng, device)


if __name__ == "__main__":
    main()
