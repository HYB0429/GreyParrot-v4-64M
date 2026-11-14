# -*- coding: utf-8 -*-
"""质量优先中文 SFT 混合数据 → GreyParrot SFT 训练二进制（v3）。

配方（v2 基础上扩充，全部经过质量筛选）：
  1. COIG-CQIA 全量（人工核验中文源）
  2. Qwen3-235B 蒸馏（去数学/STEM）
  3. MiniMind 抽样
  4. Magpie-Qwen2-Pro-200K-Chinese（instruct_reward>=3 且 input_quality=good 的头部样本）
  5. Alpaca-GPT4 中文 52k（GPT-4 任务型指令）
  6. smoltalk-chinese 分文件限量（多轮对话质感）
模板：每轮 "用户：{q}\n助手：" + "{a}" + EOS；loss 只算助手段。
输出 tokens.npy (N,512 int32, EOS 填充) 与 mask.npy (N,512 int32, 1=助手段)。
"""
import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
import sentencepiece as spm

EOS = 3
MAXLEN = 512
MAX_TURNS = 6

MATH_PAT = re.compile(
    r"\\\(|\\\)|\\\[|\\\]|\\frac|\\boxed|\\sqrt|\\sum|\\int|\\alpha|\\beta|\\pi\b"
    r"|\bx\^2\b|解方程|求导|不定积分|定积分|数列|等差|等比|三角函数|抛物线"
    r"|双曲线|椭圆方程|极坐标|排列组合|二项式|复数.*模|概率为|证明.*对任意"
)

# 推诿型 / 身份暴露型回答 → 直接丢弃（小模型学这些会快速劣化）
DODGE_PAT = re.compile(
    r"截至目前|无法直接提供|我的知识截止|训练数据截止"
    r"|作为一个(?:语言)?模型|作为AI|作为一个人工智能"
    r"|我是(?:一个)?(?:语言)?模型|我是AI助手"
    r"|由OpenAI(?:开发|训练)?|ChatGPT|GPT-4|OpenAI 开发"
    r"|[Mm]ini[Mm]ind|智能助手[小—，是]|个人开发者|官方文档"
)


def is_bad_answer(a: str) -> bool:
    return bool(DODGE_PAT.search(a))


def build_segments(conv):
    """user/assistant 交替轮 → [(prompt, response)]；含 system/tool 的样本返回 None。"""
    segs = []
    pending_user = None
    for turn in conv:
        role = turn.get("role")
        content = turn.get("content") or ""
        if role == "system" or role == "tool" or turn.get("tool_calls"):
            return None
        if role == "user":
            if pending_user is not None:
                return None
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                return None
            segs.append((pending_user, content))
            pending_user = None
    if pending_user is not None or not segs:
        return None
    return segs[:MAX_TURNS]


def rows_from_segments(segs, sp, check_bad=True):
    ids, mask = [], []
    for q, a in segs:
        if check_bad and is_bad_answer(a):
            return None
        p_ids = sp.encode(f"用户：{q}\n助手：")
        r_ids = sp.encode(a) + [EOS]
        if len(r_ids) < 3:
            return None
        ids += p_ids + r_ids
        mask += [0] * len(p_ids) + [1] * len(r_ids)
        if len(ids) > MAXLEN:
            return None
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cqia", default="/workspace/data/raw/coig_cqia/COIG-CQIA-full.jsonl")
    ap.add_argument("--distill", default="/workspace/data/raw/qwen3_distill/qwen3_235b_2507_distill_110k.jsonl")
    ap.add_argument("--minimind", default="/workspace/data/sft/sft_t2t_mini.jsonl")
    ap.add_argument("--magpie", default="/workspace/data/raw/magpie_zh/data")
    ap.add_argument("--alpaca", default="/workspace/data/raw/alpaca_gpt4_zh/train.csv")
    ap.add_argument("--smoltalk", default="/workspace/data/raw/smoltalk_zh")
    ap.add_argument("--identity", default="/workspace/data/raw/greyparrot_identity.jsonl")
    ap.add_argument("--tokenizer", default="/workspace/data/greyparrot_sp32k.model")
    ap.add_argument("--out", default="/workspace/data/sft_bin_v3")
    ap.add_argument("--distill-cap", type=int, default=60000)
    ap.add_argument("--minimind-cap", type=int, default=25000)
    ap.add_argument("--magpie-cap", type=int, default=60000)
    ap.add_argument("--alpaca-cap", type=int, default=50000)
    ap.add_argument("--smoltalk-cap", type=int, default=30000)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    pools = {}  # source -> list of (ids, mask)

    # ---- CQIA：instruction/input/output ----
    rows = []
    n_math = n_bad = 0
    with open(args.cqia, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            q = (d.get("instruction") or "").strip()
            inp = (d.get("input") or "").strip()
            a = (d.get("output") or "").strip()
            if not q or not a:
                continue
            if is_bad_answer(a):
                n_bad += 1
                continue
            if inp:
                q = q + "\n" + inp
            if MATH_PAT.search(q) or MATH_PAT.search(a):
                n_math += 1
                continue
            r = rows_from_segments([(q, a)], sp)
            if r is not None:
                rows.append(r)
    print(f"CQIA kept {len(rows):,} (math {n_math:,}, bad {n_bad:,})", flush=True)
    pools["cqia"] = rows

    # ---- Qwen3 蒸馏：messages，去数学/STEM 后封顶 ----
    rows = []
    n_math = 0
    with open(args.distill, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            msgs = d.get("messages")
            if not isinstance(msgs, list):
                continue
            segs = build_segments(msgs)
            if segs is None:
                continue
            joined = " ".join(q + " " + a for q, a in segs)
            if MATH_PAT.search(joined):
                n_math += 1
                continue
            r = rows_from_segments(segs, sp)
            if r is not None:
                rows.append(r)
    print(f"distill kept {len(rows):,} before cap (math filtered {n_math:,})", flush=True)
    if len(rows) > args.distill_cap:
        pick = rng.permutation(len(rows))[:args.distill_cap]
        rows = [rows[i] for i in sorted(pick)]
    pools["distill"] = rows
    print(f"distill after cap {len(rows):,}", flush=True)

    # ---- MiniMind 抽样 ----
    rows = []
    with open(args.minimind, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            conv = d.get("conversations")
            if not isinstance(conv, list):
                continue
            segs = build_segments(conv)
            if segs is None:
                continue
            r = rows_from_segments(segs, sp)
            if r is not None:
                rows.append(r)
    if len(rows) > args.minimind_cap:
        pick = rng.permutation(len(rows))[:args.minimind_cap]
        rows = [rows[i] for i in sorted(pick)]
    pools["minimind"] = rows
    print(f"minimind after cap {len(rows):,}", flush=True)

    # ---- Magpie-Qwen2-Pro-200K-Chinese：reward/质量头部 ----
    import pandas as pd
    rows = []
    n_drop = 0
    files = sorted(
        os.path.join(args.magpie, f)
        for f in os.listdir(args.magpie)
        if f.endswith(".parquet")
    )
    cand = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = df[(df["instruct_reward"] >= 1.0) & (df["input_quality"].isin(["good", "excellent"]))]
        cand.append(df[["instruction", "response", "instruct_reward"]])
    df = pd.concat(cand, ignore_index=True)
    # 按 reward 降序取 cap，优先高质量
    df = df.sort_values("instruct_reward", ascending=False).head(args.magpie_cap * 3)
    for _, r in df.iterrows():
        q = str(r["instruction"]).strip()
        a = str(r["response"]).strip()
        if not q or not a or is_bad_answer(a):
            n_drop += 1
            continue
        row = rows_from_segments([(q, a)], sp)
        if row is None:
            n_drop += 1
            continue
        rows.append(row)
        if len(rows) >= args.magpie_cap:
            break
    print(f"magpie kept {len(rows):,} (dropped {n_drop:,})", flush=True)
    pools["magpie"] = rows

    # ---- Alpaca-GPT4 中文：instruction/input/output ----
    rows = []
    n_bad = 0
    df = pd.read_csv(args.alpaca)
    df = df.sample(frac=1.0, random_state=args.seed)
    for _, r in df.iterrows():
        q = str(r.get("instruction") or "").strip()
        inp = str(r.get("input") or "").strip()
        a = str(r.get("output") or "").strip()
        if not q or not a or is_bad_answer(a):
            n_bad += 1
            continue
        if inp and inp.lower() != "nan":
            q = q + "\n" + inp
        if MATH_PAT.search(q) or MATH_PAT.search(a):
            continue
        row = rows_from_segments([(q, a)], sp)
        if row is None:
            continue
        rows.append(row)
        if len(rows) >= args.alpaca_cap:
            break
    print(f"alpaca kept {len(rows):,} (bad {n_bad:,})", flush=True)
    pools["alpaca"] = rows

    # ---- smoltalk-chinese：多轮，分文件限量 ----
    rows = []
    skip_files = {"math23k_zh_fixed.parquet", "math23k_zh.parquet"}
    st_files = sorted(
        f for f in os.listdir(args.smoltalk)
        if f.endswith(".parquet") and f not in skip_files
    )
    per_file = max(1, args.smoltalk_cap // max(1, len(st_files)))
    for f in st_files:
        df = pd.read_parquet(os.path.join(args.smoltalk, f))
        kept_f = 0
        for _, r in df.iterrows():
            conv = r["conversations"]
            if not isinstance(conv, (list, np.ndarray)):
                continue
            conv = [dict(c) for c in conv]
            segs = build_segments(conv)
            if segs is None:
                continue
            row = rows_from_segments(segs, sp)
            if row is None:
                continue
            rows.append(row)
            kept_f += 1
            if kept_f >= per_file or len(rows) >= args.smoltalk_cap:
                break
        if len(rows) >= args.smoltalk_cap:
            break
    print(f"smoltalk kept {len(rows):,}", flush=True)
    pools["smoltalk"] = rows

    # ---- GreyParrot 自我认知（自建数据，豁免推诿/身份过滤）----
    rows = []
    if os.path.exists(args.identity):
        with open(args.identity, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                msgs = d.get("messages")
                if not isinstance(msgs, list):
                    continue
                segs = build_segments(msgs)
                if segs is None:
                    continue
                r = rows_from_segments(segs, sp, check_bad=False)
                if r is not None:
                    rows.append(r)
    print(f"identity kept {len(rows):,}", flush=True)
    pools["identity"] = rows

    # ---- 全局去重 + 合并 ----
    seen = set()
    kept_rows = []
    dup = 0
    for src, rows in pools.items():
        kept = 0
        for ids, mask in rows:
            q = sp.decode([t for t, m in zip(ids, mask) if m == 0])
            h = hashlib.md5(q.encode()).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            kept_rows.append((ids, mask))
            kept += 1
        print(f"{src}: {kept:,} after global dedup", flush=True)

    order = rng.permutation(len(kept_rows))
    val_idx = set(order[:args.val].tolist())
    train_rows = [kept_rows[i] for i in order if i not in val_idx]
    val_rows = [kept_rows[i] for i in order[:args.val]]

    def pack(rows):
        n = len(rows)
        tokens = np.full((n, MAXLEN), EOS, dtype=np.int32)
        mask = np.zeros((n, MAXLEN), dtype=np.int32)
        for i, (ids, m) in enumerate(rows):
            tokens[i, :len(ids)] = ids
            mask[i, :len(m)] = m
        return tokens, mask

    os.makedirs(args.out, exist_ok=True)
    tr_tok, tr_msk = pack(train_rows)
    va_tok, va_msk = pack(val_rows)
    np.save(f"{args.out}/sft_train_tokens.npy", tr_tok)
    np.save(f"{args.out}/sft_train_mask.npy", tr_msk)
    np.save(f"{args.out}/sft_val_tokens.npy", va_tok)
    np.save(f"{args.out}/sft_val_mask.npy", va_msk)

    lens = np.asarray([len(ids) for ids, _ in train_rows])
    report = dict(
        sources={k: len(v) for k, v in pools.items()},
        global_dup=dup,
        train=len(train_rows), val=len(val_rows),
        len_mean=float(lens.mean()), len_p50=int(np.percentile(lens, 50)),
        len_p90=int(np.percentile(lens, 90)), len_p99=int(np.percentile(lens, 99)),
        masked_token_frac=float(tr_msk.mean()),
        minutes=round((time.time() - t0) / 60, 1),
    )
    with open(f"{args.out}/sft_prep_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
