# -*- coding: utf-8 -*-
"""A/B 生成对比：SFT 聊天模板提示。用法: python3 ab_gen.py <ckpt> [--max-new 120]"""
import argparse, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sentencepiece as spm
from model_v6_fast import build_64m, EOS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"=== checkpoint: step={ckpt.get('step')} best_val={ckpt.get('best_val')} ===")

    model, _cfg, _n = build_64m(verbose=False)
    model.load_state_dict(ckpt["model"])
    model = model.float().eval()

    sp = spm.SentencePieceProcessor(model_file=str(HERE / "checkpoints" / "greyparrot_sp32k.model"))

    prompts = [
        "用户：你好，请介绍一下你自己。\n助手：",
        "用户：用 Python 写一个快速排序，只给代码。\n助手：",
        "用户：给我讲一个简短的笑话。\n助手：",
        "用户：What is the capital of France?\n助手：",
        "用户：3+5 等于几？只回答数字。\n助手：",
    ]
    ids = [sp.encode(p) for p in prompts]
    maxlen = max(len(i) for i in ids)
    pad = [i + [EOS] * (maxlen - len(i)) for i in ids]
    idx = torch.tensor(pad, dtype=torch.long)

    t0 = time.time()
    out = model.generate(idx, args.max_new, temperature=args.temp, top_p=args.top_p, eos=EOS, use_cache=True)
    dt = time.time() - t0
    print(f"gen {args.max_new} tokens x {len(prompts)} prompts in {dt:.1f}s ({len(prompts)*args.max_new/dt:.0f} tok/s)\n")
    for p, o in zip(prompts, out):
        gen = sp.decode([t for t in o[maxlen:].tolist() if t != EOS])
        print(f"【{p}】")
        print(gen.strip()[:600])
        print("-" * 60)

if __name__ == "__main__":
    main()
