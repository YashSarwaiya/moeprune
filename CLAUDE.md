# Project: Domain-specific expert pruning of Kimi K3

## Goal

Find out how many of Kimi K3's 896 experts can be deleted before the model breaks — specifically when we only care about **one domain** (e.g. coding) rather than general capability.

Nobody has published this. K3's weights were released 27 July 2026.

## Why this question and not another

Three related problems exist. Two are closed:

| Problem | Status |
|---|---|
| Run K3's real weights on consumer hardware at all | Solved — llama.cpp + large external SSD + patience |
| Run K3 fast on a 16GB laptop | Impossible — hardware-limited, see "Constraints" |
| Shrink K3 enough to matter | **Open** — this project |

Do not propose solutions to the first two. They are settled.

## Target model facts

- **Kimi K3**, Moonshot AI. Released 16 July 2026; open weights 27 July 2026
- 2.8T total parameters, ~104B active per token
- **896 experts, 16 active per token** (Stable LatentMoE) — the most fine-grained public MoE
- Architecture: Kimi Delta Attention (KDA) + Attention Residuals (AttnRes)
- 1M token context, native vision
- **MXFP4 weights / MXFP8 activations**, quantization-aware training from the SFT stage onward
- Always-on thinking; `reasoning_effort` = low/high/max, default max
- License: modified MIT ("Kimi K3 License")
- Full checkpoint: 1.56 TB across 96 shards
- Requires vLLM, or llama.cpp via Unsloth's PR fork (mainline lacks KDA support)

### Existing quantizations (Unsloth)

| Build | Size | Retained accuracy |
|---|---|---|
| UD-Q8_K_XL | 1.56 TB | lossless |
| UD-Q4_K_XL | ~1.51 TB | near-lossless |
| UD-IQ2 | 861 GB | ~90% |
| UD-IQ1_S | 594 GB | ~78.9% |

**Important:** post-training quantization is largely spent. K3 shipped natively at 4-bit because of QAT, so the usual 4× compression dividend was already applied before release. Compression must come from architecture, not bit-width.

## Prior art

### General expert pruning — the current ceiling

- **HEAPr** (arXiv 2509.22299) — Hessian-based, near-lossless at 20–25% compression on DeepSeek MoE and Qwen MoE
- **MoNE** (arXiv 2507.00390) — max pruning within 1% accuracy loss: 25% on Qwen2-57B-A14B, 24% on Qwen3-30B-A3B. Notes that *larger models tolerate more aggressive pruning*
- **MoE-I²** (arXiv 2411.01016) — 25% expert pruning is effectively lossless; >50% expert parameter reduction achievable when combined with intra-expert low-rank decomposition plus finetuning
- **MoE-Pruner** (arXiv 2410.12013) — Mixtral-8x7B at 50% sparsity retains 99% of original performance after expert-wise knowledge distillation
- **SlimQwen** (arXiv 2605.08738) — deeper compression, but done during pretraining with continual training + KD. Datacenter-scale

**Ceiling of one-shot general pruning: ~25%. With retraining: ~50%.**

### The lever this project uses

- **Domain-specific pruning with few-shot demonstrations** (arXiv 2504.06792) — introduces *few-shot expert localization*: domain-specific experts can be reliably identified using only a handful of demonstrations. Explicitly notes that models with many fine-grained experts (e.g. 256/layer) may be more specialized than coarse ones like Mixtral

This is the core method. Domain-restricted pruning should permit far higher ratios than general pruning, because capability outside the target domain is deliberately sacrificed.

### What experts actually specialize in (caveats)

The naive "one expert per subject" model is wrong. Evidence is mixed:

- Mixtral's experts route largely by **syntax and token type**, not subject matter. The term "expert" is arguably misleading
- **What Gets Activated** (arXiv 2601.10159) — only a minority of experts show strong domain specialization; most show minimal domain-specific behavior
- **Illusion of Specialization** (arXiv 2601.03425) — a small set of disproportionately active "super experts" does much of the work; most experts are polysemous, and routing aligns only weakly with human semantic domains
- **Do Domain-specific Experts exist in MoE-based LLMs?** (arXiv 2604.05267) — tested 10 MoE models (3.8B–120B) and found empirical evidence that domain-specific experts *do* exist
- Shared vs routed split: shared experts act as generalists, routed experts refine domain-specific attributes

**Implication:** super experts and shared experts must be preserved regardless of domain. Only the long tail is prunable. Do not assume clean topic partitioning.

## Hypothesis

K3's 896-expert pool is more prunable than any model tested so far, because:

1. Specialization increases with expert granularity, and 896 is unprecedented
2. Larger models have already been shown to tolerate more aggressive pruning
3. Restricting to a single domain removes the constraint that forced the ~25–50% ceiling in prior work

**Prediction to test:** a coding-only K3 retains most coding capability at pruning ratios well above 50%.

## Method sketch

1. Acquire weights (`moonshotai/Kimi-K3`) on rented multi-GPU hardware — 8×H100 80GB is the practical floor for full-precision work
2. Instrument the router. Log expert activation across a domain-specific calibration set (start with ~20 coding problems per the few-shot localization result)
3. Rank experts by contribution. Use norm-aware / output-contribution measures, not raw routing frequency — frequency conflates syntactic regularity with semantic importance
4. Identify and protect shared experts and super experts
5. Prune in a sweep: 25 / 50 / 65 / 80 / 90% of routed experts removed
6. Evaluate each point on in-domain benchmarks (coding) and out-of-domain (to confirm the tradeoff is real and measured, not accidental)
7. Optional recovery pass: expert-wise knowledge distillation from unpruned K3, which recovered Mixtral to 99% at 50% sparsity
8. Publish the quality-vs-ratio curve

The deliverable is **the curve**, not a working local model.

## Constraints

**Development hardware:** MacBook Pro M1 Pro, 16 GB unified memory. Thunderbolt 4 external storage, ~3 GB/s practical.

This machine cannot run K3 in any form. The binding numbers:

- ~22 GB of weights must be read per forward pass at 1-bit
- ~3 GB/s storage bandwidth → ~7s/token floor before any overhead
- The M1 Pro GPU (~10 TFLOPS FP16) is too slow to batch large speculative-decoding trees, which is the technique that rescues offloaded inference elsewhere (SpecExec, arXiv 2406.02532, reaches 4–6 tok/s for 70B on consumer GPUs; SubSpec, arXiv 2509.18344, reaches 25 tok/s for 7B in 8GB VRAM)
- Always-on max-effort thinking multiplies token count before any visible output

**Therefore:** rented GPUs for pruning and any full-model inference. The laptop is for orchestration, analysis, and evaluating small pruned artifacts only.

**Even at the best proven pruning ratio, K3 will not fit in 16 GB.** 50% of 594 GB is ~300 GB. This project produces a research result that benefits 128–300 GB machines, not a local chatbot.

## Out of scope

- Making K3 interactive on a 16 GB laptop — hardware-limited, not a software problem
- Further post-training quantization as the primary compression lever — QAT already spent it
- Distillation into a small dense model — valid and mature, but a different project. (For a working local assistant today, use an existing distilled model such as Qwen 3.5 9B at ~7 GB.)

## Open questions

- Does specialization keep increasing past 256 experts, or plateau?
- How much of the 896 is "standing committee" that can never be cut?
- Does KDA's hybrid linear attention change which experts matter per layer?
- Is the quality-vs-ratio curve smooth or does it collapse at a threshold?
- Can domain-pruned variants be recombined, or does that undo the savings?

## How to help

Useful: router instrumentation code, expert-contribution scoring methods, evaluation harness design, cost estimates for rented GPU sweeps, critiques of the hypothesis.

Not useful: suggestions to run K3 on the laptop, or to quantize harder.
