# -*- coding: utf-8 -*-
"""GreyParrot v6：64M Dense / TeLU / RKM｜固定预算 TSA-BLR。

训练：loss, aux = model(idx, targets, ce_chunk=256)
推理：logits, aux = model(idx, last_logits_only=True)
分块交叉熵在 backward 重算 logits，不保存完整词表激活。
"""
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core_v6 import VEC_ELEM_BUDGET
from core_v6_fast import RMSNormFast as RMSNorm
from core_v6_fast import ParallelDualBrainV6Fast as ParallelDualBrainV6
from core_v6_fast import DenseFFNV6Fast as DenseFFNV6

__all__ = ["DEFAULT_CFG_V6", "V6Block", "GreyParrotV6",
           "build_64m", "fit_d_ff", "count_params", "VOCAB", "EOS"]
VOCAB = 32018
EOS = 3

DEFAULT_CFG_V6 = dict(
    d_model=640, n_layers=8, n_heads=10, head_dim=64, d_ff=1536,
    rkm_modes=4, rkm_heads=4, rkm_ctx_max=64.0, rkm_conv_k=1,
    rkm_trapz=True, rkm_phase=True, rkm_omega_base=True,
    rkm_noncompetitive=False, rkm_scan_chunk=64,
    tsa_m=8, tsa_a=4, tsa_r=2, tsa_w_loc=8, tsa_seg=32,
    tsa_d_c=16, tsa_delta=0.1, tsa_vec_budget=VEC_ELEM_BUDGET,
    tsa_top_b=4, tsa_d_router=32,
    dropout=0.0, activation_checkpoint=False, ce_chunk=256,
    report_stats=False,
)


class V6Block(nn.Module):
    def __init__(self, cfg, layer_idx, directions):
        super().__init__()
        D = cfg["d_model"]
        self.idx = layer_idx
        self.dual = ParallelDualBrainV6(
            D, H=cfg["n_heads"], hd=cfg["head_dim"], directions=directions,
            rkm_modes=cfg["rkm_modes"], rkm_heads=cfg["rkm_heads"],
            rkm_ctx_max=cfg["rkm_ctx_max"], rkm_conv_k=cfg["rkm_conv_k"],
            rkm_trapz=cfg["rkm_trapz"], rkm_phase=cfg["rkm_phase"],
            rkm_omega_base=cfg["rkm_omega_base"],
            rkm_noncompetitive=cfg["rkm_noncompetitive"],
            a=cfg["tsa_a"], r=cfg["tsa_r"], w_loc=cfg["tsa_w_loc"],
            seg=cfg["tsa_seg"], d_c=cfg["tsa_d_c"], delta=cfg["tsa_delta"],
            vec_budget=cfg["tsa_vec_budget"], top_b=cfg["tsa_top_b"],
            d_router=cfg["tsa_d_router"], report_stats=cfg["report_stats"])
        self.dual.rkm.scan_chunk = int(cfg.get("rkm_scan_chunk", 64))
        self.ln_ffn = RMSNorm(D)
        self.ffn = DenseFFNV6(D, cfg["d_ff"])
        self.gamma_ffn = nn.Parameter(torch.ones(D) * 0.5)

    def forward(self, x, disable=None, chunk=None):
        xa, aux = self.dual(x, disable=disable, chunk=chunk)
        return xa + self.gamma_ffn * self.ffn(self.ln_ffn(xa)), aux


class _ChunkedCrossEntropy(torch.autograd.Function):
    """按 token 分块计算 CE；反向重算每块 logits，梯度等价于标准 mean CE。"""
    @staticmethod
    def forward(ctx, h, targets, weight, chunk, ignore_index):
        B, T, D = h.shape
        hf = h.reshape(B * T, D)
        tf = targets.reshape(-1)
        chunk = max(1, min(int(chunk), hf.shape[0]))
        device_type = h.device.type
        autocast_on = torch.is_autocast_enabled(device_type)
        compute_dtype = (torch.get_autocast_dtype(device_type) if autocast_on else h.dtype)
        ctx.compute_dtype = compute_dtype
        n_valid = (tf != ignore_index).sum()
        loss_sum = torch.zeros((), device=h.device, dtype=torch.float32)
        wc = weight.to(compute_dtype)
        for i in range(0, hf.shape[0], chunk):
            hc, tc = hf[i:i + chunk], tf[i:i + chunk]
            hc_compute = hc.to(compute_dtype)
            lp = F.log_softmax(F.linear(hc_compute, wc).float(), dim=-1)
            valid = tc != ignore_index
            safe = torch.where(valid, tc, torch.zeros_like(tc))
            nll = -lp.gather(1, safe.unsqueeze(1)).squeeze(1)
            loss_sum += (nll * valid.to(nll.dtype)).sum()
        ctx.save_for_backward(h, targets, weight, n_valid)
        ctx.chunk, ctx.ignore_index = chunk, ignore_index
        return loss_sum / n_valid.clamp_min(1).to(loss_sum.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        h, targets, weight, n_valid = ctx.saved_tensors
        B, T, D = h.shape
        hf, tf = h.reshape(B * T, D), targets.reshape(-1)
        gh_flat = torch.empty_like(hf)
        gw = torch.zeros_like(weight)
        denom = n_valid.clamp_min(1).to(torch.float32)
        wc = weight.to(ctx.compute_dtype)
        for i in range(0, hf.shape[0], ctx.chunk):
            hc, tc = hf[i:i + ctx.chunk], tf[i:i + ctx.chunk]
            compute_dtype = ctx.compute_dtype
            hc_compute = hc.to(compute_dtype)
            lp = F.log_softmax(F.linear(hc_compute, wc).float(), dim=-1)
            valid = tc != ctx.ignore_index
            safe = torch.where(valid, tc, torch.zeros_like(tc))
            grad_logits = lp.exp()
            grad_logits.scatter_add_(1, safe.unsqueeze(1),
                                     -valid.to(grad_logits.dtype).unsqueeze(1))
            grad_logits *= valid.to(grad_logits.dtype).unsqueeze(1)
            grad_logits *= (grad_output.float() / denom)
            gd = grad_logits.to(compute_dtype)
            gh_flat[i:i + ctx.chunk] = torch.matmul(gd, wc).to(h.dtype)
            gw.add_(torch.matmul(gd.transpose(0, 1), hc_compute).to(weight.dtype))
        return gh_flat.view_as(h), None, gw, None, None


def chunked_cross_entropy(h, targets, weight, chunk=256, ignore_index=-100):
    return _ChunkedCrossEntropy.apply(h, targets, weight, chunk, ignore_index)


class GreyParrotV6(nn.Module):
    def __init__(self, V=VOCAB, cfg=None):
        super().__init__()
        self.cfg = dict(DEFAULT_CFG_V6 if cfg is None else cfg)
        c = self.cfg
        assert c["n_heads"] * c["head_dim"] == c["d_model"]
        assert c["tsa_a"] <= c["tsa_m"]
        D, L = c["d_model"], c["n_layers"]
        self.V, self.D, self.n_layers = V, D, L
        self.embed = nn.Embedding(V, D)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.drop = nn.Dropout(c["dropout"])
        self.register_buffer("directions",
                             torch.randn(c["tsa_m"], c["head_dim"]) / math.sqrt(c["head_dim"]))
        self.blocks = nn.ModuleList([V6Block(c, i, self.directions) for i in range(L)])
        self.ln_f = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.embed.weight
        scale = (2.0 * L) ** -0.5
        for blk in self.blocks:
            for proj in (blk.dual.rkm.Wo, blk.dual.tsa.Wo, blk.ffn.down):
                with torch.no_grad():
                    proj.weight.mul_(scale)

    def forward_hidden(self, idx, disable=None, chunk=None, activation_checkpoint=None):
        x = self.drop(self.embed(idx))
        aux = x.new_zeros(())
        use_ckpt = self.cfg["activation_checkpoint"] if activation_checkpoint is None else activation_checkpoint
        for blk in self.blocks:
            if use_ckpt and self.training and torch.is_grad_enabled():
                x, a = checkpoint(blk, x, disable=disable, chunk=chunk,
                                  use_reentrant=False)
            else:
                x, a = blk(x, disable=disable, chunk=chunk)
            aux = aux + a
        return self.ln_f(x), aux

    def forward(self, idx, targets=None, disable=None, chunk=None,
                return_hidden=False, last_logits_only=False, ce_chunk=None,
                activation_checkpoint=None, ignore_index=-100):
        h, aux = self.forward_hidden(idx, disable=disable, chunk=chunk,
                                    activation_checkpoint=activation_checkpoint)
        if targets is not None:
            cc = self.cfg["ce_chunk"] if ce_chunk is None else int(ce_chunk)
            if cc and cc > 0:
                loss = chunked_cross_entropy(h, targets, self.head.weight, cc, ignore_index)
            else:
                logits = self.head(h)
                loss = F.cross_entropy(logits.float().reshape(-1, self.V),
                                       targets.reshape(-1), ignore_index=ignore_index)
            return (loss, aux, h) if return_hidden else (loss, aux)
        if last_logits_only:
            h = h[:, -1:]
        logits = self.head(h)
        return (logits, aux, h) if return_hidden else (logits, aux)

    @torch.no_grad()
    def parallel_report(self):
        rows = []
        for i, blk in enumerate(self.blocks):
            d = blk.dual
            mass = d.tsa.last_mass
            rows.append(dict(
                layer=i,
                norm_L=round(d.last_norm_L, 5) if d.last_norm_L is not None else None,
                norm_R=round(d.last_norm_R, 5) if d.last_norm_R is not None else None,
                kept=round(d.tsa.last_kept, 4) if d.tsa.last_kept is not None else None,
                pool=d.tsa.last_pool, single_shot=d.tsa.last_vec,
                mass_dir=round(mass[0], 4) if mass else None,
                mass_loc=round(mass[1], 4) if mass else None,
                mass_shared=round(mass[2], 4) if mass else None,
                aux=0.0))
        return rows

    gate_report = parallel_report

    @torch.no_grad()
    def prefill_cached(self, idx, max_len):
        """整段 prefill，返回最后 logits 和逐层 RKM/TSA 缓存。"""
        self.eval()
        x = self.embed(idx)
        caches = []
        for blk in self.blocks:
            x, cache = blk.dual.prefill(x, max_len)
            x = x + blk.gamma_ffn * blk.ffn(blk.ln_ffn(x))
            caches.append(cache)
        return self.head(self.ln_f(x[:, -1:])), caches

    @torch.no_grad()
    def step_cached(self, idx, caches):
        """输入一个已采样 token，返回它的下一 token logits。"""
        if idx.shape[1] != 1:
            raise ValueError("step_cached 一次只接受一个 token")
        x = self.embed(idx)
        for blk, cache in zip(self.blocks, caches):
            x = blk.dual.step(x, cache)
            x = x + blk.gamma_ffn * blk.ffn(blk.ln_ffn(x))
        return self.head(self.ln_f(x))

    @staticmethod
    def _sample(logits, temperature, top_p):
        lg = logits[:, -1, :].float() / max(1e-4, temperature)
        if top_p < 1.0:
            probs = torch.softmax(lg, -1)
            sp, si = torch.sort(probs, descending=True)
            sp = sp.masked_fill(sp.cumsum(-1)-sp > top_p, 0.0)
            sp = sp/sp.sum(-1, keepdim=True).clamp_min(1e-8)
            return torch.gather(si, 1, torch.multinomial(sp, 1))
        return torch.multinomial(torch.softmax(lg, -1), 1)

    @torch.no_grad()
    def generate(self, idx, max_new=40, temperature=0.7, top_p=0.9,
                 eos=EOS, min_len=4, max_seq_len=512, use_cache=True):
        self.eval()
        if max_seq_len < 1:
            raise ValueError("max_seq_len 必须为正整数")
        result = idx
        finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        if use_cache:
            context = idx[:, -max_seq_len:]
            logits, caches = self.prefill_cached(context, context.shape[1] + max_new)
        for i in range(max_new):
            if not use_cache:
                logits, _ = self(result[:, -max_seq_len:], last_logits_only=True)
            nxt = self._sample(logits, temperature, top_p)
            if eos is not None:
                # 已结束样本只继续写入 EOS，不再生成任意后续内容。
                nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos), nxt)
            result = torch.cat([result, nxt], dim=1)
            generated = i + 1
            if eos is not None and generated >= min_len:
                finished |= nxt.squeeze(1).eq(eos)
                if bool(finished.all()):
                    break
            if use_cache and i + 1 < max_new:
                logits = self.step_cached(nxt, caches)
        return result


def count_params(model):
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            total += p.numel()
    return total


def fit_d_ff(V=VOCAB, target=64_000_000, cfg=None, step=8, verbose=False):
    base = dict(DEFAULT_CFG_V6 if cfg is None else cfg)
    d0, d1 = 256, 256 + step
    c0, c1 = dict(base), dict(base)
    c0["d_ff"], c1["d_ff"] = d0, d1
    n0 = count_params(GreyParrotV6(V, c0))
    n1 = count_params(GreyParrotV6(V, c1))
    slope = (n1 - n0) / step
    raw = d0 + (target - n0) / slope
    dff = max(step, min(8192, int(round(raw / step) * step)))
    c = dict(base); c["d_ff"] = dff
    n = count_params(GreyParrotV6(V, c))
    if verbose:
        print(f"N(d_ff) = {n0:,} + {slope:,.0f} * (d_ff - {d0})")
        print(f"目标 {target:,} -> d_ff {raw:.1f} -> 取整 {dff} -> {n:,}")
    best = (abs(n - target), dff, n)
    for cand in (dff - step, dff + step):
        if step <= cand <= 8192:
            cc = dict(base); cc["d_ff"] = cand
            nn_ = count_params(GreyParrotV6(V, cc))
            if abs(nn_ - target) < best[0]:
                best = (abs(nn_ - target), cand, nn_)
    return best[1], best[2]


def build_64m(V=VOCAB, target=64_000_000, cfg=None, verbose=False):
    dff, n = fit_d_ff(V, target, cfg, verbose=verbose)
    c = dict(DEFAULT_CFG_V6 if cfg is None else cfg)
    c["d_ff"] = dff
    return GreyParrotV6(V, c), c, n


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, cfg, n = build_64m(verbose=True)
    print(f"参数量 {n:,} ({n/1e6:.3f}M)，d_ff={cfg['d_ff']}，TSA 池={model.blocks[0].dual.tsa.pool_size}")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_v6.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(vocab=VOCAB, eos=EOS, params=n, cfg=cfg), f,
                  ensure_ascii=False, indent=2)
    print("配置已写入", out)
