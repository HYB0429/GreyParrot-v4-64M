# GreyParrot-v4-64M

![logo](logo.png)

A 64M-parameter Chinese/English conversational language model trained entirely from scratch by a single developer. Architecture, training code and data recipes are all open and reproducible.

- **Parameters**: 63,979,160 (~64M)
- **Architecture**: hybrid **RKM** multi-scale selective state-space + **TSA** fixed-pool sparse attention — *not* a vanilla Transformer
- **Training hardware**: a single AMD Radeon W7900D (RDNA3 / gfx1100, 48GB), ROCm + PyTorch 2.6
- **Weights**: bf16 single file, ~169 MB
- **Developer**: hyb0429
- **Positioning**: a minimal conversational model for tinkering and research. **It is not a productivity tool** — knowledge, reasoning and stability are all bounded by the 64M scale. See [Limitations](#limitations).

## Quick start

```python
import torch, sentencepiece as spm
from model_v6_fast import build_64m, EOS

model, cfg, n_params = build_64m(verbose=False)
ckpt = torch.load("greyparrot_64m_chat_v0.1_bf16.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model = model.float().eval()

sp = spm.SentencePieceProcessor(model_file="greyparrot_sp32k.model")
prompt = "用户：你好，请介绍一下你自己。\n助手："   # chat template: full-width colon
ids = torch.tensor([sp.encode(prompt)])

out = model.generate(ids, 120, temperature=0.7, top_p=0.9, eos=EOS, use_cache=True)
print(sp.decode(out[0].tolist()))
```

Chat template is fixed as `用户：{question}\n助手：` (note the full-width colon `：`), EOS id = 3. More test prompts in `code/ab_gen.py`.

> v0.1 is the SFT-v1 final checkpoint (step 8612, val 3.2149). A small identity patch (self-introduction data) is planned as v0.1.1 — those weights currently live on a halted training machine and will be attached when it is back.

Generation runs at ~80-100 tok/s on CPU (measured in this repo's dev sandbox); the whole model fits comfortably on any consumer GPU.

## Weights

The v0.1 chat checkpoint is attached to the [v0.1 release](../../releases/download/v0.1/greyparrot_64m_chat_v0.1_bf16.pt) (169 MB, bf16 weights-only: `model` / `cfg` / `step`). It exceeds GitHub's 100 MB file limit, so it is distributed as a release asset rather than in the git tree.

## Architecture

| Item | Value |
|---|---|
| Vocab | 32,018 (SentencePiece BPE) |
| d_model | 640 |
| Layers | 8 |
| Attention heads | 10 x head_dim 64 |
| FFN | d_ff 1600, output RMSNorm (ln_offn) |

Every block combines two heterogeneous branches:

**RKM (multi-scale selective state-space model)** — modes=4, heads=4, scan_chunk=64, trapezoidal parameterization + phase term + omega_base frequency basis. Handles long sequences at linear complexity.

**TSA (fixed-pool sparse attention)** — pool=45, top_b=4, delta=0.1, seg=32, adaptive query chunking 1-3. A fixed-size memory pool replaces the full KV cache: long-range retrieval with constant inference memory.

Engineering: chunked-scan forward *and* backward with merged kernel launches; adaptive TSA query chunking (1-3) so short and long sequences both avoid launch overhead; `torch.compile` target `hidden-notsa` (RKM/FFN fully compiled, TSA kept eager due to data-dependent control flow).

## Training pipeline

1. **Pretraining** on `hyb0429/greyparrot-v3-mixed-385m` (ModelScope, mixed zh/en corpus), multiple time-boxed windows, to step 71,186 — val_loss 2.3683.
2. **SFT v1** (`greyparrot64m_sft`): MiniMind sft_t2t 150k, 2h window, step 8,612, val_loss 3.2149. Learned chat format (code blocks, lists, Chinese retention); content was the weak layer.
3. **SFT v2** (`greyparrot64m_sft_v2`): quality-first 77,160-sample mix (COIG-CQIA + Qwen3-235B distill minus math + MiniMind), 5 epochs. val plateaued at 3.59 while open-ended generation collapsed — see Pitfall #2. Not released.
4. **SFT v3** (`greyparrot64m_sft_v3`): broad 166,328-sample mix (Magpie-zh, Alpaca-GPT4-zh, smoltalk-zh added, plus 41 self-identity samples) restarted from pretraining; halted after discovering v2's weights were poisoned.
5. **v0.1 release** = SFT v1 final. An identity patch (41 "I am GreyParrot" samples + 3,000 v3-mix samples, lr 5e-6, 1 epoch) is queued as v0.1.1 so the model introduces itself correctly; it was trained but its weights are on the (currently halted) training box.

### Data recipe (v3 mix, builder in `code/prep_sft_v3.py`)

| Source | Samples | Processing |
|---|---|---|
| COIG-CQIA (m-a-p) | 25,135 | human-verified Chinese, math removed |
| Qwen3-235B-2507 distill | 30,820 | math/STEM removed |
| MiniMind sft_t2t | 25,000 | sampled |
| Magpie-Qwen2-Pro-200K-Chinese | 22,330 | reward>=1.0, quality good/excellent |
| Alpaca-GPT4 Chinese | 47,186 | GPT-4 task instructions |
| smoltalk-chinese | 20,590 | per-file caps, math files excluded |
| GreyParrot self-identity (self-built) | 41 | fixed persona: 64M / RKM+TSA / hyb0429 |

Uniform processing: refusal/persona-leak regex filter, math-length filter, global MD5 dedup on first user turn, drop (never truncate) samples over 512 tokens, loss on assistant spans only.

## Reproduction

```bash
# Pretraining data: hyb0429/greyparrot-v3-mixed-385m on ModelScope
python train_v6_fast.py ...            # see run configs in docs

# SFT (--init = weights only, --resume = full state)
python train_sft.py --data-dir sft_bin_v3 --init <pretrain_weights> \
  --device cuda --dtype bf16 --batch-size 16 --lr 2e-5 \
  --max-steps 20800 --compile --compile-target hidden-notsa
```

Environment: ROCm 6.x, PyTorch 2.6 (note: `torch.load` defaults to `weights_only=True` since 2.6 — loading checkpoints from this repo needs `weights_only=False`), sentencepiece, numpy.

## Pitfalls (all actually hit)

Ranked by pain:

1. **Never-validated save path**. SFT ran 2,000 steps before the first checkpoint save, which crashed on an undefined `val_rng` — two hours of training lost. *Always smoke-test the save/restore path first.*
2. **Narrow-distribution over-training collapse** (the big one). v2 aggressively filtered out "AI-assistant-style" answers and trained 5 epochs. In-distribution val stayed flat at 3.59 from step 10,000 — while open-ended generation rotted into "number of numbers" loops. *In-distribution validation cannot detect generative degeneration; run open-ended A/B generation regularly.*
3. **ROCm host-side deadlock**. Log stops advancing, step counter frozen, yet the process burns CPU (one thread spinning at 97%, GPU busy=0). Kill and resume from the latest checkpoint. Detection heuristic: log mtime >5 min stale with frozen step count.
4. **`pkill -f` self-kill**. `pkill -f "train_sft.py"` also matches the wrapping remote shell and kills your own SSH session (exit 255). Use `pgrep -af` + `kill <PID>`.
5. **`setsid nohup` background launch hangs SSH**. The session doesn't exit until local timeout — the process *did* start. *Always verify in a fresh connection; never retry the launch after a timeout*, or two automation paths start duplicate trainers writing the same run dir (happened with v3).
6. **torch 2.6 `weights_only=True` default**. Loading own checkpoints (containing a `TorchVersion` object) fails unless you pass `weights_only=False`.
7. **Full-width-paren regex disaster**. A filter written as `智能助手（小|—|，|是）` intended a group, but full-width（）are literal characters and the inner `|` is still alternation — the branch `，` matched nearly every Chinese sentence and silently killed 99% of the data. *Always use half-width non-capturing groups `(?:...)`.*
8. **rocm-smi measurement side effect**. Querying GPU state stalls training (reproducibly). Keep monitoring scripts off high-frequency rocm-smi calls.
9. **Shared multi-tenant box**. Neighbors at 100% GPU and load averages of 15+ are normal there; check your own process before blaming the machine.
10. **SentencePiece decode normalizes punctuation**. Full-width `，？：` decode to half-width `,?:`. Normalize both sides before matching dataset text, or audits silently miss rows.
11. **Parallel-scp split parts arrive incomplete**. Always verify byte counts before `cat`, or the reassembled checkpoint blows up at `torch.load` time.
12. **Mid-cosine checkpoints are not judgeable**. Un-annealed mid-schedule checkpoints produce systematically distorted samples (v2 step-10000 was once misdiagnosed as "data poisoning"; it was just the schedule). Final quality calls require the annealed final.

## Optimization log

- **Pretraining**: 46,037 → 71,186 steps across time-boxed windows, val 2.649 → 2.368 (best 2.3683 @ 71186). Throughput 16.9k tok/s on the W7900D (bf16, B16xT512, log basis).
- **Kernel-launch merging**: chunked-scan fwd/bwd launch merging; adaptive TSA query chunking 1-3 so neither short nor long sequences waste launches.
- **Compile strategy**: `--compile-target hidden-notsa` — RKM/FFN fully compiled via inductor, TSA's data-dependent control flow stays eager. Stable.
- **Restart-free LR reschedule**: the LambdaLR closure reads `args.max_steps` live; when 8 epochs proved too many at step 16,000, resuming with `--max-steps 24000` re-anchored the cosine mid-flight (verified: 6.4e-6 matched prediction) — no restart needed.
- **Data iteration**: v1 (single source) → v2 (quality trio + heavy filtering, 5 epochs, collapse) → v3 (six-source broad mix + self-identity) → v0.1 (v1 + identity patch). Every version was A/B'd at temperature 0.7 *and* 0.1.
- **Checkpoint slimming**: training checkpoints with optimizer state are 768 MB; the slim script casts to bf16 weights-only at 169 MB, then 8-way parallel transfer with byte-count verification (see Pitfall #11).

## Limitations

- 64M params: small knowledge base, frequent factual errors, weak math/complex reasoning — verify anything important.
- Refusal-style responses ("我无法提供…") and occasional degenerate loops; higher sampling temperature makes both worse.
- No internet access; knowledge ends at the training-data cutoff.
- Multi-turn and long-context quality degrade quickly (trained at T=512).
- v0.1 has no self-identity training: asked "who are you", it dodges or mispersonates. v0.1.1 will fix this.

## Roadmap

- [ ] Full v3 recipe from pretraining (166k six-source mix, 2 epochs, lr 2e-5)
- [ ] Stronger multi-turn (sharegpt-style long dialogs, larger T)
- [ ] DPO experiment
- [ ] Quantized builds (int4/int8, CPU-friendly)

## Repo layout

- `model_v6_fast.py` (repo root) — model definition (RKM + TSA)
- `code/train_sft.py` — SFT trainer (init/resume, chunked CE, LambdaLR)
- `code/prep_sft_v3.py` — v3 data builder (sources, filters, dedup, packing)
- `code/gen_identity.py` — self-identity data generator
- `code/ab_gen.py` — A/B generation harness
- `docs/ENGINEERING.md` — v4 engineering notes: chunked scans, fused ops, cache correctness, benchmarks (Chinese)
- `config.json` — model config
- `greyparrot_sp32k.model` — SentencePiece tokenizer
- `greyparrot_64m_chat_v0.1_bf16.pt` — v0.1 weights (release asset, see above)

## Acknowledgements

- **AMD**: all training was done on a single AMD Radeon W7900D (RDNA3, 48GB) with ROCm and PyTorch. Thanks to the AMD ROCm team for the open software stack that makes consumer-grade GPU training accessible to individual developers.
- **The Mamba team**: thanks to Albert Gu, Tri Dao and the selective state space model community (S4, Mamba, and follow-ups). The multi-scale selective scan design of the RKM branch is directly inspired by their work.
- **The open data ecosystem**: COIG-CQIA, MiniMind, Magpie, Alpaca-GPT4, smoltalk-chinese and the Qwen team, whose public datasets and distills made the SFT mixture possible.

## License

Code and weights: Apache-2.0. Training data comes from public datasets (COIG-CQIA, Qwen3 distill, MiniMind, Magpie, Alpaca-GPT4, smoltalk-chinese, etc.) under their respective licenses — verify each source's terms before commercial use.
