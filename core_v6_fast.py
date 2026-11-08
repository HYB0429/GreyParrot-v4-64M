# -*- coding: utf-8 -*-
"""GreyParrot v6 fast：可融合 TeLU-GLU + 精确 RKM/TSA 增量缓存。
权重名称与 core_v6 完全一致；缓存路径仅用于 eval/no_grad 推理。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_v6 import (RMSNorm, RKMExpertV6, TSASkeletonV6, _normalize,
                     _rotate_cs, _affine_scan, apply_rope, telu, VEC_ELEM_BUDGET)

__all__ = ["RKMExpertV6Fast", "TSASkeletonV6Fast", "ParallelDualBrainV6Fast",
           "DenseFFNV6Fast", "FusedTeLUGLU", "FusedRMSNorm", "RMSNormFast"]


class FusedRMSNorm(torch.autograd.Function):
    """RMSNorm 单 autograd 节点；缓存 x 与 rstd，合并反向归约表达式。"""
    @staticmethod
    def forward(ctx, x, weight, eps):
        xf = x.float()
        rstd = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return (xf * rstd * weight.float()).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, weight, rstd = ctx.saved_tensors
        xf = x.float()
        gf = grad.float()
        gx = gf * weight.float()
        dx = rstd * gx - xf * rstd.pow(3) * (gx * xf).mean(-1, keepdim=True)
        axes = tuple(range(grad.ndim - 1))
        dw = (gf * xf * rstd).sum(dim=axes) if axes else gf * xf * rstd
        return dx.to(x.dtype), dw.to(weight.dtype), None


class RMSNormFast(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return FusedRMSNorm.apply(x, self.w, self.eps)


class FusedTeLUGLU(torch.autograd.Function):
    """单 autograd 节点的 TeLU(gate)*up；backward 一次返回两部分梯度。
    用 torch.compile 包装调用点可进一步融合其中的逐元素算子。
    """
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return telu(gate) * up

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        gf = gate.float()
        gc = gf.clamp(-20.0, 15.0)
        e = torch.exp(gc)
        th = torch.tanh(e)
        # clamp 区间内包含指数项导数；区间外指数饱和，但外层 x 的导数仍保留。
        active = (gf >= -20.0) & (gf <= 15.0)
        extra = torch.where(active, gf * e * (1.0 - th.square()),
                            torch.zeros_like(th))
        derivative = th + extra
        dgate = grad.float() * up.float() * derivative
        dup = grad.float() * (gf * th)
        return dgate.to(gate.dtype), dup.to(up.dtype)


class DenseFFNV6Fast(nn.Module):
    def __init__(self, D, d_ff):
        super().__init__()
        self.d_ff = int(d_ff)
        self.gate_up = nn.Linear(D, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).split(self.d_ff, dim=-1)
        return self.down(FusedTeLUGLU.apply(gate, up))


def _scan_pair(a, b, initial=None):
    """同时返回 inclusive 累计乘积 a 和状态 b。"""
    if initial is not None:
        b = b.clone()
        b[:, 0] = b[:, 0] + a[:, 0].unsqueeze(-1)*initial
    off = 1
    while off < a.shape[1]:
        ao, bo = a, b
        a = ao.clone()
        b = bo.clone()
        a[:, off:] = ao[:, off:]*ao[:, :-off]
        b[:, off:] = bo[:, off:] + ao[:, off:].unsqueeze(-1)*bo[:, :-off]
        off *= 2
    return a, b


def _affine_scan_chunked(A, L, state=None, chunk_size=64):
    """块内并行扫描 + 块摘要扫描 + 合并；短序列直接使用原扫描。"""
    B, T = A.shape[:2]
    C = min(int(chunk_size), T)
    if T <= C:
        return _affine_scan(A, L, state=state)
    N = (T + C - 1)//C
    pad = N*C-T
    if pad:
        A = torch.cat([A, A.new_ones(B, pad, *A.shape[2:])], 1)
        L = torch.cat([L, L.new_zeros(B, pad, *L.shape[2:])], 1)
    Af = A.reshape(B*N, C, *A.shape[2:])
    Lf = L.reshape(B*N, C, *L.shape[2:])
    a_loc, b_loc = _scan_pair(Af, Lf)
    a_tot = a_loc[:, -1].reshape(B, N, *A.shape[2:])
    b_tot = b_loc[:, -1].reshape(B, N, *L.shape[2:])
    _, b_inc = _scan_pair(a_tot, b_tot, initial=state)
    first_b = torch.zeros_like(b_inc[:, :1]) if state is None else state.unsqueeze(1)
    prefix_b = torch.cat([first_b, b_inc[:, :-1]], 1)
    a_loc = a_loc.reshape(B, N, C, *A.shape[2:])
    b_loc = b_loc.reshape(B, N, C, *L.shape[2:])
    out = b_loc + a_loc.unsqueeze(-1)*prefix_b.unsqueeze(2)
    return out.reshape(B, N*C, *L.shape[2:])[:, :T]


class RKMExpertV6Fast(RKMExpertV6):
    def forward(self, x, state=None, return_state=False):
        B, T, D = x.shape
        M, H, dh, nP = self.M, self.H, self.dh, self.n_pairs
        xi = x
        if self.conv is not None:
            k = self.conv.kernel_size[0]
            xi = self.conv(F.pad(x.transpose(1, 2), (k-1, 0))).transpose(1, 2)
        Vt, g_, amp_, dd_, lam_, om_ = self.W_in(xi).split(self.sizes, dim=-1)
        g = torch.sigmoid(g_).float()
        amp = 2.0*torch.sigmoid(amp_).float()/M if self.noncompetitive else torch.softmax(amp_, -1).float()
        theta = self.theta.float().view(1, 1, M, H) + dd_.float().view(B, T, M, H)
        A = torch.exp(-F.softplus(theta))
        Vh = Vt.float().reshape(B, T, H, dh)
        v = g.unsqueeze(-1).unsqueeze(-1)*Vh.unsqueeze(2)
        if self.use_trapz:
            lam = torch.sigmoid(lam_.float()).view(B, T, M, H, 1)
            v_prev = torch.cat([v.new_zeros(B, 1, M, H, dh), v[:, :-1]], dim=1)
            v = lam*v + (1.0-lam)*A.unsqueeze(-1)*v_prev
        if self.use_phase:
            omega = om_.float().view(B, T, H, nP)
            if self.omega_base is not None:
                omega = omega + self.omega_base.float().view(1, 1, H, nP)
            phi = omega.cumsum(dim=1).unsqueeze(2)
            c, s = torch.cos(phi), torch.sin(phi)
            u = _affine_scan_chunked(A, _rotate_cs(v, c, s, inverse=True), state=state,
                                     chunk_size=getattr(self, "scan_chunk", 64))
            u = _rotate_cs(u, c, s)
        else:
            u = _affine_scan_chunked(A, v, state=state, chunk_size=getattr(self, "scan_chunk", 64))
        final_state = u[:, -1]
        uhat = u*torch.rsqrt(u.square().mean(-1, keepdim=True)+1e-6)
        w = self.w_read.float().view(M, H, dh)
        wg = self.w_gate.float().view(1, 1, M, H, dh)
        gate = torch.sigmoid((uhat*wg).sum(-1))
        coeff = amp.view(B, T, M, 1)*gate
        acc = torch.einsum('btmh,btmhd,mhd->bthd', coeff, uhat, w).reshape(B, T, D)
        out = self.Wo(telu(acc.to(x.dtype)))
        return (out, final_state) if return_state else out

    def _project(self, x):
        B, T, D = x.shape
        M, H, dh = self.M, self.H, self.dh
        Vt, g_, amp_, dd_, lam_, om_ = self.W_in(x).split(self.sizes, dim=-1)
        g = torch.sigmoid(g_).float()
        amp = (2.0 * torch.sigmoid(amp_).float() / M if self.noncompetitive
               else torch.softmax(amp_, -1).float())
        theta = self.theta.float().view(1, 1, M, H) + dd_.float().view(B, T, M, H)
        A = torch.exp(-F.softplus(theta))
        raw = g.unsqueeze(-1).unsqueeze(-1) * Vt.float().reshape(B, T, H, dh).unsqueeze(2)
        omega = None
        if self.use_phase:
            omega = om_.float().view(B, T, H, self.n_pairs)
            if self.omega_base is not None:
                omega = omega + self.omega_base.float().view(1, 1, H, self.n_pairs)
        return A, raw, amp, lam_, omega

    @torch.no_grad()
    def build_cache(self, x, final_state):
        if self.conv is not None:
            raise NotImplementedError("当前增量缓存支持默认 conv_k=1")
        A, raw, amp, lam_, omega = self._project(x)
        phi = omega.cumsum(dim=1).unsqueeze(2) if self.use_phase else None
        cache = dict(raw_prev=raw[:, -1], phase_prev=phi[:, -1, 0] if phi is not None else None)
        if self.use_phase:
            c, s = torch.cos(phi[:, -1]), torch.sin(phi[:, -1])
            cache["state_rot"] = _rotate_cs(final_state.float(), c, s, inverse=True)
        else:
            cache["state_rot"] = final_state.float()
        return cache

    @torch.no_grad()
    def step(self, x, cache):
        B, T, D = x.shape
        if T != 1:
            raise ValueError("RKM step 一次只接受一个 token")
        A, raw, amp, lam_, omega = self._project(x)
        v = raw
        if self.use_trapz:
            lam = torch.sigmoid(lam_.float()).view(B, 1, self.M, self.H, 1)
            v = lam * raw + (1.0 - lam) * A.unsqueeze(-1) * cache["raw_prev"].unsqueeze(1)
        state_rot = cache["state_rot"]
        if self.use_phase:
            # 统一保存 [B,H,nP]，避免 batch 维在增量步骤中交叉广播。
            phi = cache["phase_prev"] + omega[:, 0]
            c, s = torch.cos(phi[:, None, None]), torch.sin(phi[:, None, None])
            rotated = _rotate_cs(v, c, s, inverse=True)
            state_rot = A[:, 0].unsqueeze(-1) * state_rot + rotated[:, 0]
            u = _rotate_cs(state_rot, c[:, 0], s[:, 0]).unsqueeze(1)
            cache["phase_prev"] = phi
        else:
            state_rot = A[:, 0].unsqueeze(-1) * state_rot + v[:, 0]
            u = state_rot.unsqueeze(1)
        cache["state_rot"] = state_rot
        cache["raw_prev"] = raw[:, 0]
        uhat = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + 1e-6)
        w = self.w_read.float().view(self.M, self.H, self.dh)
        wg = self.w_gate.float().view(1, 1, self.M, self.H, self.dh)
        gate = torch.sigmoid((uhat * wg).sum(-1))
        coeff = amp.view(B, 1, self.M, 1) * gate
        acc = torch.einsum('btmh,btmhd,mhd->bthd', coeff, uhat, w).reshape(B, 1, D)
        return self.Wo(telu(acc.to(x.dtype)))


class TSASkeletonV6Fast(TSASkeletonV6):
    @torch.no_grad()
    def _project_all(self, x, offset=0):
        B, T, _ = x.shape
        H, hd, BH = self.H, self.hd, B * self.H
        q_, gate_, c = self.W_qgv(x).split([self.Hhd, self.Hhd, self.d_c], dim=-1)
        k_, v_ = self.W_kuv(c).split([self.Hhd, self.Hhd], dim=-1)
        def bt(z):
            return z.reshape(B, T, H, hd).transpose(1, 2).reshape(BH, T, hd)
        q, k, v = bt(q_), bt(k_), bt(v_)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self._rope(offset + T, x.device)
        cos, sin = cos[:, offset:offset+T].to(q.dtype), sin[:, offset:offset+T].to(q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        ds = torch.matmul(k, _normalize(self.directions.to(q.dtype)).transpose(0, 1))
        qr, kr = self.W_router(x).float().split(self.d_router, dim=-1)
        return q, k, v, ds, qr, kr, gate_

    def _make_block(self, cache, block_idx):
        seg = self.seg
        i0, i1 = block_idx * seg, (block_idx + 1) * seg
        B, BH = cache["B"], cache["BH"]
        q = cache["q"][:, i0:i1]
        k, v = cache["k"][:, i0:i1], cache["v"][:, i0:i1]
        ds = cache["ds"][:, i0:i1]
        summary = _normalize(cache["kr"][:, i0:i1].mean(1))
        land = ds.reshape(BH, seg, self.m).permute(0, 2, 1).topk(self.r, dim=-1).indices
        Ql = self.land_norm(q.mean(1))
        cs = torch.einsum('bld,bd->bl', k, Ql) / math.sqrt(self.hd)
        ca = torch.softmax(cs.float(), dim=1).to(k.dtype)
        Vsh = torch.einsum('bl,bld->bd', ca, v)
        cache["summaries"].append(summary)
        cache["land"].append(land)
        cache["Ksh"].append(Ql)
        cache["Vsh"].append(Vsh)

    @torch.no_grad()
    def build_cache(self, x, max_len):
        B, T, _ = x.shape
        if max_len < T:
            raise ValueError(f"TSA cache max_len={max_len} 小于 prompt 长度 {T}")
        BH = B * self.H
        q, k, v, ds, qr, kr, _ = self._project_all(x)
        def alloc_like(z):
            shape = (z.shape[0], max_len) + tuple(z.shape[2:])
            out = z.new_empty(shape)
            out[:, :T] = z
            return out
        cache = dict(B=B, BH=BH, length=T, q=alloc_like(q), k=alloc_like(k),
                     v=alloc_like(v), ds=alloc_like(ds), qr=alloc_like(qr), kr=alloc_like(kr),
                     summaries=[], land=[], Ksh=[], Vsh=[])
        for bi in range(T // self.seg):
            self._make_block(cache, bi)
        return cache

    @torch.no_grad()
    def step(self, x, cache):
        B, T, _ = x.shape
        if T != 1:
            raise ValueError("TSA step 一次只接受一个 token")
        H, hd, BH = self.H, self.hd, B * self.H
        a, r, m, w, b = self.a, self.r, self.m, self.w_loc, self.top_b
        pos = cache["length"]
        if B != cache["B"]:
            raise ValueError("TSA step batch 与缓存不一致")
        if pos >= cache["q"].shape[1]:
            raise RuntimeError(f"TSA cache capacity exceeded: max_len={cache['q'].shape[1]}")
        q, k, v, ds, qr, kr, gate_ = self._project_all(x, offset=pos)
        for name, z in (("q", q), ("k", k), ("v", v), ("ds", ds)):
            cache[name][:, pos] = z[:, 0]
        cache["qr"][:, pos] = qr[:, 0]
        cache["kr"][:, pos] = kr[:, 0]
        cache["length"] = pos + 1
        if cache["length"] % self.seg == 0:
            self._make_block(cache, cache["length"] // self.seg - 1)
        # 刚由当前 token 完成的块仍属于当前块，必须到下一 token 才可见。
        S = min(len(cache["summaries"]), pos // self.seg)
        dt = q.dtype
        Dn = _normalize(self.directions.to(dt))
        topa = torch.matmul(q, Dn.transpose(0, 1)).topk(a, dim=-1).indices
        qr = _normalize(qr).unsqueeze(1).expand(B, H, 1, self.d_router).reshape(BH, 1, self.d_router)
        if S:
            summaries = torch.stack(cache["summaries"], 1)[:, :S]
            summaries = summaries.unsqueeze(1).expand(B, H, S, self.d_router).reshape(BH, S, self.d_router)
            route_raw = torch.bmm(qr, summaries.transpose(1, 2))
            route_pad = F.pad(route_raw, (0, max(0, b-S)), value=-torch.inf)
            blk = route_pad.topk(b, dim=-1).indices
            blk_safe = blk.clamp_max(S-1)
            route_sel = F.pad(route_raw, (0, max(0, b-S))).gather(2, blk_safe)
            valid_blk = blk < S
            land = torch.stack(cache["land"], 1)[:, :S]
            # 块内局部地标索引转绝对位置。
            block_offset = blk_safe.view(BH, b, 1, 1) * self.seg
            selected_abs = land.gather(1, blk_safe.view(BH, b, 1, 1).expand(BH, b, m, r)) + block_offset
            dir_idx = selected_abs.gather(2, topa.view(BH, 1, a, 1).expand(BH, b, a, r)).reshape(BH, 1, b*a*r)
            valid_dir = valid_blk.unsqueeze(-1).expand(BH, 1, b, a*r).reshape(BH, 1, b*a*r)
        else:
            route_sel = qr.new_zeros(BH, 1, b)
            valid_blk = torch.zeros(BH, 1, b, dtype=torch.bool, device=x.device)
            dir_idx = torch.zeros(BH, 1, b*a*r, dtype=torch.long, device=x.device)
            valid_dir = torch.zeros(BH, 1, b*a*r, dtype=torch.bool, device=x.device)
        loc = pos - torch.arange(w, -1, -1, device=x.device)
        valid_loc = loc >= 0
        loc = loc.clamp_min(0).view(1, 1, self.Aloc).expand(BH, 1, self.Aloc)
        act = torch.cat([dir_idx, loc], -1)
        valid = torch.cat([valid_dir, valid_loc.view(1, 1, self.Aloc).expand(BH, 1, self.Aloc)], -1)
        act_key = act.masked_fill(~valid, pos+1)
        act_s, order = act_key.sort(dim=-1)
        valid_s = valid.gather(-1, order)
        first = torch.ones_like(valid_s)
        first[..., 1:] = act_s[..., 1:] != act_s[..., :-1]
        valid_s = valid_s & first
        hidx = torch.arange(BH, device=x.device) % H
        bias = q.new_zeros(BH, 1, self.Aidx, dtype=torch.float32)
        bias[..., -self.Aloc:] = self.rel_bias[hidx].float().unsqueeze(1)
        bias = bias.gather(-1, order)
        route_direct = route_sel.unsqueeze(-1).expand(BH, 1, b, a*r).reshape(BH, 1, b*a*r)
        route_idx_bias = F.pad(route_direct, (0, self.Aloc)).gather(-1, order)
        route_bias_all = torch.cat([route_idx_bias, route_sel], -1) * self.route_bias
        gather_idx = act_s.clamp(0, pos)
        flat_gather = gather_idx.squeeze(1).unsqueeze(-1).expand(BH, self.Aidx, hd)
        Kidx = cache["k"].gather(1, flat_gather).unsqueeze(1)
        Vidx = cache["v"].gather(1, flat_gather).unsqueeze(1)
        if S:
            Ksh = torch.stack(cache["Ksh"], 1)[:, :S]
            Vsh = torch.stack(cache["Vsh"], 1)[:, :S]
            gi = blk_safe.unsqueeze(-1).expand(BH, 1, b, hd)
            Kcomp = Ksh.unsqueeze(1).expand(BH, 1, S, hd).gather(2, gi)
            Vcomp = Vsh.unsqueeze(1).expand(BH, 1, S, hd).gather(2, gi)
            Kall, Vall = torch.cat([Kidx, Kcomp], 2), torch.cat([Vidx, Vcomp], 2)
        else:
            Kall, Vall = F.pad(Kidx, (0, 0, 0, b)), F.pad(Vidx, (0, 0, 0, b))
        scores = torch.einsum('bqd,bqad->bqa', q, Kall).float() / math.sqrt(hd)
        valid_all = torch.cat([valid_s, valid_blk], -1)
        scores = scores + F.pad(bias, (0, b)) + route_bias_all
        scores = scores.masked_fill(~valid_all, -1.0e4)
        e = torch.exp((scores-scores.amax(-1, keepdim=True)).clamp(max=0.0))*valid_all.float()
        A = self.pool_size
        if self.delta > 0:
            ed, dsr = e.sort(dim=-1, descending=True)
            cum = ed.cumsum(-1)
            ar = torch.arange(A, device=x.device).view(1, 1, A)
            nxt = F.pad(ed[..., 1:], (0, 1))
            tail_ub = (A-ar-1).clamp_min(0).float()*nxt
            ok = self.delta*cum >= (1.0-self.delta)*tail_ub
            kk = torch.where(ok, ar, torch.full_like(ar, A)).amin(-1, keepdim=True)
            keep_srt = (ed > 0) & (ar <= kk)
            keep = torch.zeros_like(keep_srt).scatter_(-1, dsr, keep_srt)
        else:
            keep = e > 0
        weights = e*keep.float()
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(1e-12)
        out = torch.einsum('bqa,bqad->bqd', weights.to(dt), Vall)
        O = out.reshape(B, H, 1, hd).transpose(1, 2).reshape(B, 1, self.Hhd)
        return self.Wo(O*torch.sigmoid(gate_))


class ParallelDualBrainV6Fast(nn.Module):
    def __init__(self, D, H, hd, directions, **kwargs):
        super().__init__()
        rkm_keys = {"rkm_modes", "rkm_heads", "rkm_ctx_max", "rkm_conv_k", "rkm_trapz",
                    "rkm_phase", "rkm_omega_base", "rkm_noncompetitive", "report_stats"}
        rk = {k: kwargs[k] for k in rkm_keys if k in kwargs}
        self.rkm = RKMExpertV6Fast(D, M=rk.get("rkm_modes", 4), n_heads=rk.get("rkm_heads", 4),
            ctx_max=rk.get("rkm_ctx_max", 64.0), conv_k=rk.get("rkm_conv_k", 1),
            use_trapz=rk.get("rkm_trapz", True), use_phase=rk.get("rkm_phase", True),
            use_omega_base=rk.get("rkm_omega_base", True),
            noncompetitive=rk.get("rkm_noncompetitive", False), report_stats=rk.get("report_stats", False))
        tsa_keys = ["a", "r", "w_loc", "seg", "d_c", "delta", "vec_budget", "top_b", "d_router", "report_stats"]
        tk = {k: kwargs[k] for k in tsa_keys if k in kwargs}
        self.tsa = TSASkeletonV6Fast(D, H=H, hd=hd, directions=directions, **tk)
        self.ln_L, self.ln_R = RMSNorm(D), RMSNorm(D)
        self.ln_oL, self.ln_oR = RMSNorm(D), RMSNorm(D)
        self.gamma_L = nn.Parameter(torch.ones(D)*0.5)
        self.gamma_R = nn.Parameter(torch.ones(D)*0.5)
        self.report_stats = bool(kwargs.get("report_stats", False))
        self.last_norm_L = self.last_norm_R = None

    def forward(self, x, disable=None, chunk=None):
        h_L = 0.0 if disable == 'L' else self.ln_oL(self.rkm(self.ln_L(x)))
        h_R = 0.0 if disable == 'R' else self.ln_oR(self.tsa(self.ln_R(x), chunk=chunk))
        return x + self.gamma_L*h_L + self.gamma_R*h_R, x.new_zeros(())

    @torch.no_grad()
    def prefill(self, x, max_len):
        xr = self.ln_L(x)
        h_L, rstate = self.rkm(xr, return_state=True)
        rcache = self.rkm.build_cache(xr, rstate)
        xt = self.ln_R(x)
        h_R = self.tsa(xt)
        tcache = self.tsa.build_cache(xt, max_len)
        out = x + self.gamma_L*self.ln_oL(h_L) + self.gamma_R*self.ln_oR(h_R)
        return out, (rcache, tcache)

    @torch.no_grad()
    def step(self, x, cache):
        rc, tc = cache
        h_L = self.ln_oL(self.rkm.step(self.ln_L(x), rc))
        h_R = self.ln_oR(self.tsa.step(self.ln_R(x), tc))
        return x + self.gamma_L*h_L + self.gamma_R*h_R
