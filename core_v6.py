# -*- coding: utf-8 -*-
"""GreyParrot v6：有界候选 TSA + 融合投影 RKM，面向 ROCm eager/bf16。"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["telu", "TeLU", "RMSNorm", "RKMExpertV6", "TSASkeletonV6",
           "ParallelDualBrainV6", "DenseFFNV6", "VEC_ELEM_BUDGET"]
VEC_ELEM_BUDGET = 25_000_000
_TELU_HI, _TELU_LO, _NEG_MASK = 15.0, -20.0, -1.0e4


def telu(x):
    dt = x.dtype
    w = x.float() if dt in (torch.bfloat16, torch.float16) else x
    return (w * torch.tanh(torch.exp(w.clamp(_TELU_LO, _TELU_HI)))).to(dt)


class TeLU(nn.Module):
    def forward(self, x):
        return telu(x)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.w.float()).to(dt)


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)


def build_rope_cache(T, hd, device=None):
    half = hd // 2
    theta = 10000.0 ** (-torch.arange(0, half, dtype=torch.float32) / half)
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), theta)
    c, s = freqs.cos(), freqs.sin()
    return (c.to(device), s.to(device)) if device else (c, s)


def _affine_scan(A, L, state=None):
    """Hillis–Steele 仿射扫描；切片实现避免每轮构造 ones/zeros/cat。"""
    nT = A.shape[1]
    a, b = A, L
    if state is not None:
        b = b.clone()
        b[:, 0] = b[:, 0] + A[:, 0].unsqueeze(-1) * state
    off = 1
    while off < nT:
        a_old, b_old = a, b
        a = a_old.clone()
        b = b_old.clone()
        a[:, off:] = a_old[:, off:] * a_old[:, :-off]
        b[:, off:] = (b_old[:, off:] +
                      a_old[:, off:].unsqueeze(-1) * b_old[:, :-off])
        off *= 2
    return b


@torch.no_grad()
def _softplus_inv(y):
    return y + torch.log(-torch.expm1(-y))


def _rotate_cs(x, c, s, inverse=False):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    if inverse:
        return torch.stack([x1 * c + x2 * s,
                            -x1 * s + x2 * c], dim=-1).flatten(-2)
    return torch.stack([x1 * c - x2 * s,
                        x1 * s + x2 * c], dim=-1).flatten(-2)


class RKMExpertV6(nn.Module):
    """多尺度选择性 SSM；所有输入投影合并为一次 GEMM。"""
    def __init__(self, D, M=4, n_heads=4, ctx_max=64.0, conv_k=1,
                 use_trapz=True, use_phase=True, use_omega_base=True,
                 noncompetitive=False, report_stats=False):
        super().__init__()
        assert D % n_heads == 0 and (D // n_heads) % 2 == 0
        self.D, self.M, self.H, self.dh = D, M, n_heads, D // n_heads
        self.n_pairs = self.dh // 2
        self.use_trapz, self.use_phase = bool(use_trapz), bool(use_phase)
        self.noncompetitive, self.report_stats = noncompetitive, report_stats
        self.conv = None
        if conv_k and conv_k > 1:
            self.conv = nn.Conv1d(D, D, conv_k, groups=D, padding=0, bias=False)
            with torch.no_grad():
                w = torch.zeros(conv_k); w[-1] = 1.0
                self.conv.weight.copy_(w.repeat(D, 1, 1))
        n_lam = M * n_heads if self.use_trapz else 0
        n_om = n_heads * self.n_pairs if self.use_phase else 0
        self.sizes = [D, M, M, M * n_heads, n_lam, n_om]
        self.W_in = nn.Linear(D, sum(self.sizes))
        with torch.no_grad():
            self.W_in.weight.normal_(0, D ** -0.5)
            self.W_in.bias.zero_()
            p0 = D + 2 * M
            self.W_in.weight[p0:p0 + M * n_heads].zero_()
            self.W_in.weight[p0 + M * n_heads:].zero_()
        self.Wo = nn.Linear(D, D, bias=False)
        self.w_read = nn.Parameter(torch.randn(M, D) * 0.1)
        self.w_gate = nn.Parameter(torch.randn(M, D) * 0.02)
        self.omega_base = (nn.Parameter(torch.zeros(n_heads, self.n_pairs))
                           if use_omega_base else None)
        ctx = torch.logspace(math.log10(2.0), math.log10(ctx_max), M)
        self.theta = nn.Parameter(_softplus_inv(1.0 / ctx).unsqueeze(-1).repeat(1, n_heads))
        self.state_rms, self.last_phase = None, None

    def forward(self, x, state=None, return_state=False):
        B, T, D = x.shape
        M, H, dh, nP = self.M, self.H, self.dh, self.n_pairs
        xi = x
        if self.conv is not None:
            k = self.conv.kernel_size[0]
            xi = self.conv(F.pad(x.transpose(1, 2), (k - 1, 0))).transpose(1, 2)
        Vt, g_, amp_, dd_, lam_, om_ = self.W_in(xi).split(self.sizes, dim=-1)
        g = torch.sigmoid(g_).float()
        if self.noncompetitive:
            amp = 2.0 * torch.sigmoid(amp_).float() / M
        else:
            amp = torch.softmax(amp_, -1).float()
        theta = self.theta.float().view(1, 1, M, H) + dd_.float().view(B, T, M, H)
        A = torch.exp(-F.softplus(theta))
        Vh = Vt.float().reshape(B, T, H, dh)
        v = g.unsqueeze(-1).unsqueeze(-1) * Vh.unsqueeze(2)
        if self.use_trapz:
            lam = torch.sigmoid(lam_.float()).view(B, T, M, H, 1)
            v_prev = torch.cat([v.new_zeros(B, 1, M, H, dh), v[:, :-1]], dim=1)
            v = lam * v + (1.0 - lam) * A.unsqueeze(-1) * v_prev
        if self.use_phase:
            omega = om_.float().view(B, T, H, nP)
            if self.omega_base is not None:
                omega = omega + self.omega_base.float().view(1, 1, H, nP)
            phi = omega.cumsum(dim=1).unsqueeze(2)
            c, s = torch.cos(phi), torch.sin(phi)
            u = _affine_scan(A, _rotate_cs(v, c, s, inverse=True), state=state)
            u = _rotate_cs(u, c, s)
            if self.report_stats:
                with torch.no_grad():
                    self.last_phase = float(omega.abs().mean())
        else:
            u = _affine_scan(A, v, state=state)
        final_state = u[:, -1]
        uhat = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + 1e-6)
        w = self.w_read.float().view(M, H, dh)
        wg = self.w_gate.float().view(1, 1, M, H, dh)
        gate = torch.sigmoid((uhat * wg).sum(-1))
        coeff = amp.view(B, T, M, 1) * gate
        acc = torch.einsum('btmh,btmhd,mhd->bthd', coeff, uhat, w).reshape(B, T, D)
        out = self.Wo(telu(acc.to(x.dtype)))
        if self.report_stats:
            self.state_rms = final_state.pow(2).mean(-1).mean(dim=(0, 1)).sqrt().detach()
        return (out, final_state) if return_state else out


class TSASkeletonV6(nn.Module):
    """TSA-BLR：固定预算的块路由地标注意力。

    每个 query 的池：top_b 已完成块 × top_a 方向 × r 地标、局部窗、
    以及 top_b 个块内压缩 KV。只有完整且严格早于 query 所在块的块可见。
    路由对块摘要仍有 O(T*S) 项，但精确 K/V 主注意力为 O(T*A)，A 固定。
    """
    def __init__(self, D, H, hd, directions, a=4, r=2, w_loc=8, seg=32,
                 d_c=16, delta=0.1, vec_budget=VEC_ELEM_BUDGET,
                 top_b=4, d_router=32, report_stats=False):
        super().__init__()
        assert a <= directions.shape[0] and top_b >= 1 and 1 <= r <= seg
        self.D, self.H, self.hd = D, H, hd
        self.Hhd = H * hd
        self.directions = directions
        self.m = directions.shape[0]
        self.a, self.r, self.w_loc, self.seg = int(a), int(r), int(w_loc), int(seg)
        self.top_b = int(top_b)
        self.d_c, self.delta = int(d_c), float(delta)
        self.vec_budget = int(vec_budget)
        self.report_stats = bool(report_stats)
        self.Aloc = self.w_loc + 1
        self.Aidx = self.top_b * self.a * self.r + self.Aloc
        self.pool_size = self.Aidx + self.top_b

        self.W_qgv = nn.Linear(D, 2 * self.Hhd + self.d_c)
        with torch.no_grad():
            self.W_qgv.weight[self.Hhd:2 * self.Hhd].zero_()
            self.W_qgv.bias.zero_()
        self.W_kuv = nn.Linear(self.d_c, 2 * self.Hhd, bias=False)
        self.W_router = nn.Linear(D, 2 * d_router, bias=False)
        self.route_bias = nn.Parameter(torch.tensor(0.1))
        self.d_router = int(d_router)
        self.Wo = nn.Linear(self.Hhd, D, bias=False)
        self.q_norm, self.k_norm = RMSNorm(hd), RMSNorm(hd)
        self.land_norm = RMSNorm(hd)
        self.rel_bias = nn.Parameter(torch.zeros(H, self.Aloc))
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.last_kept = self.last_mass = self.last_vec = None
        self.last_pool = self.pool_size

    def _rope(self, T, device):
        if (self.rope_cos is None or self.rope_cos.shape[0] < T
                or self.rope_cos.device != device):
            c, s = build_rope_cache(T, self.hd, device)
            self.rope_cos, self.rope_sin = c, s
        return self.rope_cos[:T].unsqueeze(0), self.rope_sin[:T].unsqueeze(0)

    def forward(self, x, chunk=None):
        B, T, _ = x.shape
        H, hd, Hhd, BH = self.H, self.hd, self.Hhd, x.shape[0] * self.H
        a, r, m, w, seg, b = self.a, self.r, self.m, self.w_loc, self.seg, self.top_b
        dev = x.device
        scale = 1.0 / math.sqrt(hd)

        q_, gate_, c = self.W_qgv(x).split([Hhd, Hhd, self.d_c], dim=-1)
        k_, v_ = self.W_kuv(c).split([Hhd, Hhd], dim=-1)
        # autocast 下 x 常为 fp32，而投影输出为 bf16；以投影 dtype 为准。
        dt = q_.dtype
        q_, k_, v_ = q_.to(dt), k_.to(dt), v_.to(dt)

        def bt(z):
            return z.reshape(B, T, H, hd).transpose(1, 2).reshape(BH, T, hd)

        q, k, v = bt(q_), bt(k_), bt(v_)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self._rope(T, dev)
        q = apply_rope(q, cos.to(dt), sin.to(dt))
        k = apply_rope(k, cos.to(dt), sin.to(dt))
        g_attn = torch.sigmoid(gate_)
        pos = torch.arange(T, device=dev)

        Dn = _normalize(self.directions.to(dt))
        ds = torch.matmul(k, Dn.transpose(0, 1))
        aff = torch.matmul(q, Dn.transpose(0, 1))
        topa = aff.topk(a, dim=-1).indices                 # [BH,T,a]

        # 只预计算完整块；不完整尾块永远不会成为“已完成块”。
        S = T // seg
        qr, kr = self.W_router(x).float().split(self.d_router, dim=-1)
        qr = _normalize(qr).unsqueeze(1).expand(B, H, T, self.d_router).reshape(BH, T, self.d_router)
        if S:
            summaries = _normalize(kr[:, :S * seg].reshape(B, S, seg, self.d_router).mean(2))
            summaries = summaries.unsqueeze(1).expand(B, H, S, self.d_router).reshape(BH, S, self.d_router)
            route_raw = torch.bmm(qr, summaries.transpose(1, 2))
            current_block = pos // seg
            route_valid = torch.arange(S, device=dev).view(1, S) < current_block.view(T, 1)
            route = route_raw.masked_fill(~route_valid.view(1, T, S), -torch.inf)
            qb = q[:, :S * seg].reshape(BH, S, seg, hd)
            kb = k[:, :S * seg].reshape(BH, S, seg, hd)
            vb = v[:, :S * seg].reshape(BH, S, seg, hd)
            land = ds[:, :S * seg].reshape(BH, S, seg, m).permute(0, 1, 3, 2).topk(r, dim=-1).indices
            land = land + (seg * torch.arange(S, device=dev)).view(1, S, 1, 1)
            # 每个块用自己的均值 query 压缩本块 K/V，不做全前缀 O(T*S) 压缩。
            Ql = self.land_norm(qb.mean(2))
            cs = torch.einsum('bsld,bsd->bsl', kb, Ql) * scale
            ca = torch.softmax(cs.float(), dim=2).to(dt)
            Vsh = torch.einsum('bsl,bsld->bsd', ca, vb)
            Ksh = Ql
        else:
            route_raw = route = qr.new_full((BH, T, 0), -torch.inf)
            route_valid = torch.zeros(T, 0, dtype=torch.bool, device=dev)
            land = torch.empty(BH, 0, m, r, dtype=torch.long, device=dev)
            Ksh = q.new_empty(BH, 0, hd)
            Vsh = v.new_empty(BH, 0, hd)

        # 即使 S < top_b 也返回固定 b 列；补入的块全部由 valid_blk 掩码。
        route_pad = F.pad(route, (0, max(0, b - S)), value=-torch.inf)
        blk = route_pad.topk(b, dim=-1).indices           # [BH,T,b]
        blk_safe = blk.clamp_max(max(S - 1, 0))
        if S:
            route_sel = F.pad(route_raw, (0, max(0, b - S))).gather(1, blk_safe)
            valid_blk = route_valid.unsqueeze(0).expand(BH, T, S).gather(-1, blk_safe)
            valid_blk = valid_blk & (blk < S)
            land_rows = land.unsqueeze(1).expand(BH, T, S, m, r)
            selected = land_rows.gather(2, blk_safe.view(BH, T, b, 1, 1).expand(BH, T, b, m, r))
            dir_idx = selected.gather(3, topa.view(BH, T, 1, a, 1).expand(BH, T, b, a, r))
            dir_idx = dir_idx.reshape(BH, T, b * a * r)
            valid_dir = valid_blk.unsqueeze(-1).expand(BH, T, b, a * r).reshape(BH, T, b * a * r)
        else:
            route_sel = qr.new_zeros(BH, T, b)
            valid_blk = torch.zeros(BH, T, b, dtype=torch.bool, device=dev)
            dir_idx = torch.zeros(BH, T, b * a * r, dtype=torch.long, device=dev)
            valid_dir = torch.zeros_like(valid_blk).unsqueeze(-1).expand(BH, T, b, a * r).reshape(BH, T, b * a * r)

        loc = pos.view(T, 1) - torch.arange(w, -1, -1, device=dev).view(1, -1)
        valid_loc = loc >= 0
        loc = loc.clamp_min(0).view(1, T, self.Aloc).expand(BH, T, self.Aloc)
        act = torch.cat([dir_idx, loc], dim=-1)
        valid = torch.cat([valid_dir,
                           valid_loc.view(1, T, self.Aloc).expand(BH, T, self.Aloc)], dim=-1)
        # 无效候选统一映射到哨兵 T，避免排序/去重受未来块内容影响。
        act_key = act.masked_fill(~valid, T)
        act_s, order = act_key.sort(dim=-1)
        valid_s = valid.gather(-1, order)
        first = torch.ones_like(valid_s)
        first[..., 1:] = act_s[..., 1:] != act_s[..., :-1]
        valid_s = valid_s & first

        hidx = torch.arange(BH, device=dev) % H
        bias = q.new_zeros(BH, T, self.Aidx, dtype=torch.float32)
        bias[..., -self.Aloc:] = self.rel_bias[hidx].float().unsqueeze(1)
        bias = bias.gather(-1, order)
        valid_all = torch.cat([valid_s, valid_blk], dim=-1)
        bias_all = F.pad(bias, (0, b))
        # 硬 top-k 只决定索引；所选块的连续相关性同时作为小幅 logit 偏置，
        # 因而 W_router 有实际梯度，而不是名义可训练的死参数。
        route_direct = route_sel.unsqueeze(-1).expand(BH, T, b, a * r).reshape(BH, T, b * a * r)
        route_idx_bias = F.pad(route_direct, (0, self.Aloc)).gather(-1, order)
        route_bias_all = torch.cat([route_idx_bias, route_sel], dim=-1) * self.route_bias
        bh_off = (torch.arange(BH, device=dev) * T).view(BH, 1, 1)
        gather_idx = act_s.clamp(0, T - 1) + bh_off
        kf, vf = k.reshape(BH * T, hd), v.reshape(BH * T, hd)
        A = self.pool_size

        def core(i0, cc):
            sl = slice(i0, i0 + cc)
            gi = gather_idx[:, sl].reshape(BH, cc * self.Aidx)
            Kidx = kf.index_select(0, gi.reshape(-1)).reshape(BH, cc, self.Aidx, hd)
            Vidx = vf.index_select(0, gi.reshape(-1)).reshape(BH, cc, self.Aidx, hd)
            if S:
                gs = blk_safe[:, sl]
                comp_idx = gs.unsqueeze(-1).expand(BH, cc, b, hd)
                Kcomp = Ksh.unsqueeze(1).expand(BH, cc, S, hd).gather(2, comp_idx)
                Vcomp = Vsh.unsqueeze(1).expand(BH, cc, S, hd).gather(2, comp_idx)
                Kall = torch.cat([Kidx, Kcomp], dim=2)
                Vall = torch.cat([Vidx, Vcomp], dim=2)
            else:
                Kall = F.pad(Kidx, (0, 0, 0, b))
                Vall = F.pad(Vidx, (0, 0, 0, b))
            # K/V 保持输入精度；仅 logits/softmax 上转，避免四维 fp32 K/V。
            scores = torch.einsum('bqd,bqad->bqa', q[:, sl], Kall).float() * scale
            scores = scores + bias_all[:, sl] + route_bias_all[:, sl]
            va = valid_all[:, sl]
            scores = scores.masked_fill(~va, _NEG_MASK)
            e = torch.exp((scores - scores.amax(-1, keepdim=True)).clamp(max=0.0)) * va.float()
            if self.delta > 0:
                ed, dsr = e.sort(dim=-1, descending=True)
                cum = ed.cumsum(-1)
                ar = torch.arange(A, device=dev).view(1, 1, A)
                nxt = F.pad(ed[..., 1:], (0, 1))
                tail_ub = (A - ar - 1).clamp_min(0).float() * nxt
                ok = self.delta * cum >= (1.0 - self.delta) * tail_ub
                kk = torch.where(ok, ar, torch.full_like(ar, A)).amin(-1, keepdim=True)
                keep_srt = (ed > 0) & (ar <= kk)
                keep = torch.zeros_like(keep_srt).scatter_(-1, dsr, keep_srt)
            else:
                keep = e > 0
            weights = e * keep.float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            out = torch.einsum('bqa,bqad->bqd', weights.to(dt), Vall)
            return out, keep, weights

        # 真正的 query 分块预算：chunk 由单个 query 块的 K/V 元素数解析得到。
        elems = BH * T * A * hd
        if chunk is not None:
            ch = max(1, min(T, int(chunk)))
        elif elems <= self.vec_budget:
            ch = T
        else:
            # 自动模式只用 1～3 个 query 块，避免预算紧张时产生长 Python 循环。
            # 显式 --tsa-chunk 仍可覆盖；超预算 3 倍以上时需自行降低 batch/seq。
            n_chunks = min(3, max(1, (elems + self.vec_budget - 1) // self.vec_budget))
            ch = (T + n_chunks - 1) // n_chunks
        outs, keeps, wns = [], [], []
        for i0 in range(0, T, ch):
            o, kp, wn = core(i0, min(ch, T - i0))
            outs.append(o)
            if self.report_stats:
                keeps.append(kp); wns.append(wn)
        out = torch.cat(outs, dim=1)
        self.last_vec = int(ch == T)
        if self.report_stats:
            with torch.no_grad():
                keep = torch.cat(keeps, dim=1)
                wn = torch.cat(wns, dim=1)
                self.last_kept = float(keep.float().mean())
                self.last_mass = [float(wn[..., :b*a*r].sum(-1).mean()),
                                  float(wn[..., b*a*r:self.Aidx].sum(-1).mean()),
                                  float(wn[..., self.Aidx:].sum(-1).mean())]
        O = out.reshape(B, H, T, hd).transpose(1, 2).reshape(B, T, Hhd)
        return self.Wo(O * g_attn)


class ParallelDualBrainV6(nn.Module):
    """RKM 与 TSA 全量并行直加；监控默认关闭，不产生 GPU→CPU 同步。"""
    def __init__(self, D, H, hd, directions, rkm_modes=4, rkm_heads=4,
                 rkm_ctx_max=64.0, rkm_conv_k=1, rkm_trapz=True,
                 rkm_phase=True, rkm_omega_base=True, rkm_noncompetitive=False,
                 a=4, r=2, w_loc=8, seg=32, d_c=16, delta=0.1,
                 vec_budget=VEC_ELEM_BUDGET, top_b=4, d_router=32,
                 report_stats=False):
        super().__init__()
        self.rkm = RKMExpertV6(D, M=rkm_modes, n_heads=rkm_heads,
                               ctx_max=rkm_ctx_max, conv_k=rkm_conv_k,
                               use_trapz=rkm_trapz, use_phase=rkm_phase,
                               use_omega_base=rkm_omega_base,
                               noncompetitive=rkm_noncompetitive,
                               report_stats=report_stats)
        self.tsa = TSASkeletonV6(D, H=H, hd=hd, directions=directions,
                                 a=a, r=r, w_loc=w_loc, seg=seg, d_c=d_c,
                                 delta=delta, vec_budget=vec_budget,
                                 top_b=top_b, d_router=d_router,
                                 report_stats=report_stats)
        self.ln_L, self.ln_R = RMSNorm(D), RMSNorm(D)
        self.ln_oL, self.ln_oR = RMSNorm(D), RMSNorm(D)
        self.gamma_L = nn.Parameter(torch.ones(D) * 0.5)
        self.gamma_R = nn.Parameter(torch.ones(D) * 0.5)
        self.report_stats = bool(report_stats)
        self.last_norm_L = self.last_norm_R = None

    def forward(self, x, disable=None, chunk=None):
        h_L = 0.0 if disable == 'L' else self.ln_oL(self.rkm(self.ln_L(x)))
        h_R = 0.0 if disable == 'R' else self.ln_oR(self.tsa(self.ln_R(x), chunk=chunk))
        out = x + self.gamma_L * h_L + self.gamma_R * h_R
        if self.report_stats:
            with torch.no_grad():
                self.last_norm_L = float(h_L.pow(2).mean().sqrt()) if torch.is_tensor(h_L) else 0.0
                self.last_norm_R = float(h_R.pow(2).mean().sqrt()) if torch.is_tensor(h_R) else 0.0
        return out, x.new_zeros(())


class DenseFFNV6(nn.Module):
    """TeLU-GLU；gate/up 合并为一次 GEMM。"""
    def __init__(self, D, d_ff):
        super().__init__()
        self.d_ff = int(d_ff)
        self.gate_up = nn.Linear(D, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).split(self.d_ff, dim=-1)
        return self.down(telu(gate) * up)
