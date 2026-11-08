# -*- coding: utf-8 -*-
"""GreyParrot v6 单机单卡训练；ROCm 沿用 PyTorch 的 cuda 接口。

示例：
python train_v6_fast.py --data tokens.npy --out runs/gp64 --device cuda \
  --dtype bf16 --batch-size 8 --seq-len 512 --grad-accum 1 --max-steps 10000

数据是一维整数 token 的 .npy；最后 1%（可用 --val-tokens 改）作验证集。
"""
import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from model_v6_fast import build_64m, VOCAB


def parse_args():
    p = argparse.ArgumentParser(description="GreyParrot v6 64M ROCm trainer")
    p.add_argument("--data", required=True, help="一维 token .npy，支持内存映射")
    p.add_argument("--out", default="runs/greyparrot64")
    p.add_argument("--resume", default="", help="latest.pt 路径")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=10000, help="optimizer steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ce-chunk", type=int, default=1024, help="48GB GPU 推荐 1024；OOM 时降至 512/256")
    p.add_argument("--rkm-scan-chunk", type=int, default=64, help="RKM 两级仿射扫描块长；>=seq-len 时走原扫描")
    p.add_argument("--tsa-chunk", type=int, default=0, help="0=按显存预算自动，query 循环自适应为 1～3 块")
    p.add_argument("--activation-checkpoint", action="store_true")
    p.add_argument("--compile", action="store_true", help="可选；动态 top-k/sort 可能图分割")
    p.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--max-seconds", type=float, default=0, help="0=只按 max-steps；用于限时基准")
    p.add_argument("--val-data", default="", help="可选独立一维 token .npy；不从训练尾部切验证")
    p.add_argument("--val-tokens", type=int, default=0)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-threads", type=int, default=8)
    return p.parse_args()


def capture_rng_state(train_rng, val_rng, device):
    state = dict(train_rng=train_rng.bit_generator.state,
                 val_rng=val_rng.bit_generator.state,
                 torch=torch.get_rng_state())
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path, model, optimizer, scheduler, step, best_val, cfg, args,
                    train_rng, val_rng, device, val_offsets):
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), step=step, best_val=best_val,
                   cfg=cfg, args=vars(args), torch_version=torch.__version__,
                   rng_state=capture_rng_state(train_rng, val_rng, device),
                   val_offsets=np.asarray(val_offsets))
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 cuda，但 PyTorch 未检测到 ROCm/CUDA 设备")
    use_bf16 = args.dtype == "bf16"
    if use_bf16 and device.type != "cuda":
        raise ValueError("此训练入口的 bf16 模式仅用于 ROCm/CUDA；CPU 自检请用 --dtype fp32")
    if args.batch_size < 1 or args.seq_len < 2 or args.grad_accum < 1:
        raise ValueError("batch/seq/grad-accum 参数非法")
    if args.rkm_scan_chunk < 1:
        raise ValueError("--rkm-scan-chunk 必须为正整数")
    torch.set_num_threads(max(1, args.num_threads))
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.load(args.data, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f"需要一维整数 token 数组，实际 shape={tokens.shape}, dtype={tokens.dtype}")
    if len(tokens) < args.seq_len * args.batch_size + 3:
        raise ValueError("token 数据太短")
    val_array = None
    if args.val_data:
        val_array = np.load(args.val_data, mmap_mode="r", allow_pickle=False)
        if val_array.ndim != 1 or not np.issubdtype(val_array.dtype, np.integer):
            raise ValueError("独立验证数据也必须是一维整数 token 数组")
        train_end, val_tokens = len(tokens), len(val_array)
    else:
        val_tokens = args.val_tokens or min(2_000_000, max(args.seq_len * args.batch_size + 2,
                                                          len(tokens) // 100))
        val_tokens = min(val_tokens, len(tokens) // 10)
        train_end = len(tokens) - val_tokens
    if train_end < args.seq_len + 2 or val_tokens < args.seq_len + 2:
        raise ValueError("训练/验证切分后数据不足；请减小 seq-len 或调整 --val-tokens")
    print(f"data={len(tokens):,} tokens, train={train_end:,}, val={val_tokens:,}, "
          f"dtype={tokens.dtype}, device={device}, autocast={args.dtype}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    train_rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 1)
    n = args.seq_len + 1
    val_start = 0 if val_array is not None else train_end
    val_source = val_array if val_array is not None else tokens
    val_end = len(val_source)
    val_offsets = val_rng.integers(0, val_end - val_start - n + 1,
                                   size=(args.val_batches, args.batch_size)) + val_start

    def get_batch(split, val_index=None):
        if split == "train":
            rng, source, start, end = train_rng, tokens, 0, train_end
            offsets = rng.integers(0, end - start - n + 1, size=args.batch_size) + start
        else:
            source = val_source
            offsets = val_offsets[val_index]
        arr = np.stack([np.asarray(source[o:o + n], dtype=np.int64) for o in offsets])
        pair = torch.from_numpy(arr)
        if device.type == "cuda":
            pair = pair.pin_memory()
        pair = pair.to(device, non_blocking=True)
        return pair[:, :-1], pair[:, 1:]

    model, cfg, n_params = build_64m(verbose=True)
    cfg["rkm_scan_chunk"] = args.rkm_scan_chunk
    for blk in model.blocks:
        blk.dual.rkm.scan_chunk = args.rkm_scan_chunk
    model = model.to(device)
    tsa_chunk = args.tsa_chunk or None
    print(f"parameters={n_params:,}; TSA candidates/query="
          f"{model.blocks[0].dual.tsa.pool_size}; activation_checkpoint={args.activation_checkpoint}")
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_cfg = ckpt.get("cfg", {})
        runtime_only = {"ce_chunk", "activation_checkpoint", "rkm_scan_chunk",
                        "tsa_vec_budget", "report_stats"}
        mismatched = {k: (v, cfg.get(k)) for k, v in saved_cfg.items()
                      if k not in runtime_only and k in cfg and cfg[k] != v}
        if mismatched:
            raise ValueError(f"resume 模型配置不兼容: {mismatched}")
        saved_args = ckpt.get("args", {})
        changed_training = [k for k in ("batch_size", "seq_len", "grad_accum", "lr",
                                       "min_lr", "warmup", "max_steps")
                            if k in saved_args and saved_args[k] != getattr(args, k)]
        if changed_training:
            print("warning: resume 改变了训练参数:", ", ".join(changed_training))
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, best_val = int(ckpt["step"]), float(ckpt.get("best_val", float("inf")))
        if "val_offsets" in ckpt:
            val_offsets = np.asarray(ckpt["val_offsets"])
        rng_state = ckpt.get("rng_state")
        if rng_state:
            train_rng.bit_generator.state = rng_state["train_rng"]
            val_rng.bit_generator.state = rng_state["val_rng"]
            torch.set_rng_state(rng_state["torch"])
            if device.type == "cuda" and "cuda" in rng_state:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
        else:
            print("warning: 旧 checkpoint 不含 RNG；本次续训不是严格随机状态恢复")
        print(f"resumed step={step}, best_val={best_val}")
    train_net = torch.compile(model, mode=args.compile_mode) if args.compile else model
    if args.compile:
        print("torch.compile enabled; eager fallback: remove --compile")

    def autocast():
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=use_bf16)

    @torch.no_grad()
    def validate():
        train_net.eval()
        values = []
        for val_index in range(len(val_offsets)):
            x, y = get_batch("val", val_index)
            with autocast():
                loss, _ = train_net(x, y, chunk=tsa_chunk, ce_chunk=args.ce_chunk,
                                    activation_checkpoint=False)
            values.append(loss.detach())
        train_net.train()
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
            x, y = get_batch("train")
            with autocast():
                loss, _ = train_net(x, y, chunk=tsa_chunk, ce_chunk=args.ce_chunk,
                                    activation_checkpoint=args.activation_checkpoint)
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
            nsteps = step - last_step
            n_tokens = nsteps * args.batch_size * args.seq_len * args.grad_accum
            interval_s = max(1e-9, now - last_time)
            flops_per_token = 6 * n_params + 2 * VOCAB * cfg["d_model"]
            if args.activation_checkpoint:
                flops_per_token += 2 * n_params
            record = dict(step=step, train_loss=float(running_loss / max(1, running_micro)),
                          lr=optimizer.param_groups[0]["lr"], tokens_per_s=n_tokens / interval_s,
                          elapsed_s=now-last_time, parameter_equivalent_training_tflops=n_tokens*flops_per_token/interval_s/1e12)
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
        if args.save_interval > 0 and (step % args.save_interval == 0 or step == args.max_steps):
            save_checkpoint(out / "latest.pt", model, optimizer, scheduler, step,
                            best_val, cfg, args, train_rng, val_rng, device, val_offsets)
            print("saved", out / "latest.pt", flush=True)

    if args.max_steps == 0 or args.max_seconds > 0:
        save_checkpoint(out / "latest.pt", model, optimizer, scheduler, step,
                        best_val, cfg, args, train_rng, val_rng, device, val_offsets)


if __name__ == "__main__":
    main()
