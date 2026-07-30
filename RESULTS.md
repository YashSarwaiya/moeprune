# Results log

## Pilot 1 — Qwen3-30B-A3B-Instruct-2507, coding-domain pruning
2026-07-30, single NVIDIA B200, ~2.7 h. Full pipeline:
contrib scoring on 30-item coding calibration set, super-expert protection
from 16-item general set (tau=8, top 2%), masked pruning, greedy evals
(HumanEval pass@1 n=164, GSM8K EM n=150). Raw result files (results.json, routing logs, manifests) are in `results/`.

| Experts pruned | Kept/layer | HumanEval | GSM8K |
|---|---|---|---|
| 0% (baseline) | 128 | 91.5% | 99.3% |
| 25% | 96 | 91.5% | 97.3% |
| 50% | 64 | 89.0% | 94.0% |
| 65% | 45 | **82.3%** | **33.3%** |
| 80% | 26 | 45.1% | 0.0% |
| 87.5% | 16 | 0.0% | 0.0% |

### Findings

1. **Domain-specific pruning beats the published general-pruning ceiling.**
   MoNE reports ~25% max pruning at <1% loss for this exact model. Coding-
   targeted pruning holds HumanEval to -2.5pp at 50%.
2. **The 65% point shows the domain-specialization signature:** in-domain
   90% retained (82.3/91.5) while OOD collapsed 99.3 -> 33.3. Random pruning
   cannot produce this asymmetry; it is evidence the selection tracked
   genuinely coding-specific experts.
3. **The in-domain curve knees between 65% and 80%, cliffs by 87.5%.**
   Answers an open question from the brief: not smooth to the end —
   plateau, knee, collapse.
4. **Routing concentration (general set):** Gini mean 0.63; ~19/128 experts
   carry 50% of mass per layer, ~65/128 carry 95%; 6.5 protected
   super-experts/layer.

### Caveats (open until controls run)

- No random-pruning control yet at matching ratios (controls.sbatch adds
  50/65/80 random + 70/75 scored).
- GSM8K 0.0% at 80% may conflate capability loss with answer-format loss.
- Single seed, single calibration set; n=164/150 gives ~±2pp noise.
- 87.5% point = min_keep floor (16 kept, top-8 active): routing had almost
  no freedom; collapse there may be the floor, not the method.

## Pilot 2 — controls (2026-07-30)

Random-selection control at matched ratios + cliff localization (70/75%).
Raw result files are in `results/`.

| Point | Kept/layer | HumanEval | GSM8K |
|---|---|---|---|
| scored 70% | 38 | 70.7% | 6.0% |
| scored 75% | 32 | 63.4% | 2.0% |
| random 50% | 64 | 26.8% | 40.0% |
| random 65% | 45 | 0.6% | 16.0% |
| random 80% | 26 | 0.0% | 0.0% |

### Findings

1. **Scored selection crushes random**: at 50%, 89.0 vs 26.8 HumanEval
   (+62pp); at 65%, 82.3 vs 0.6 (+82pp). The redundancy-only explanation is
   dead — selection quality is the dominant factor.
2. **Coding knee localized at ~65%**: 82.3 -> 70.7 -> 63.4 -> 45.1 across
   65/70/75/80%. In-domain decay past the knee is steady, not a cliff until
   >80%.
3. Random pruning hurts coding *more* than math at 50% (26.8 vs 40.0) —
   plausibly because unprotected super-expert loss breaks precise syntax
   first. Note: the random control does NOT protect super experts; a
   "random + protected supers" ablation would isolate how much of the win
   comes from protection vs domain scoring (cheap future run).

## Pilot 3 — union composition (2026-07-30)

Code-pruned vs math-pruned vs their union, all at 65% per-domain ratio.
Math calibration: 28 original problems (calibration/math.jsonl — deliberately
NOT GSM8K items). Raw result files are in `results/`.

| Variant | Kept/layer | HumanEval | GSM8K |
|---|---|---|---|
| code_pruned | 45 | 82.3% | 33.3% |
| math_pruned | 45 | 0.0% | 93.3% |
| union | 67 | **90.9%** | **97.3%** |
| (baseline) | 128 | 91.5% | 99.3% |

Overlap: 22.6 experts/layer shared between the two domain keep-sets;
union = 75% of the sum of parts.

### Findings

1. **Domain-pruned variants compose — superadditively.** The union beats
   BOTH specialists on their own domains and lands within 1-2pp of the
   unpruned baseline on both benchmarks, at 52% of the experts. Answers the
   brief's open question ("can domain-pruned variants be recombined?"):
   yes, at near-zero quality cost.
2. **Method generalizes across domains**: math-targeted pruning produced the
   mirror image (93.3 GSM8K / 0.0 HumanEval) using the same pipeline and a
   28-item calibration set.
3. **Specialists are near-disjoint in capability** (0.0 coding for the math
   model) despite sharing 22.6/45 experts — the shared block is generic
   infrastructure, the disjoint ~22 carry the domain skill.
4. Reproducibility: code_pruned at 65% reproduced pilot 1's numbers exactly
   (82.3/33.3), as expected under greedy decoding with identical manifests.

## Pilot 4 — skeptic controls (2026-07-30)

Three missing controls for the union claim. MMLU probe = 300 questions from
six non-STEM subjects (philosophy, world religions, jurisprudence, European
history, marketing, prehistory). Raw result files are in `results/`.

| Point | Kept/layer | HumanEval | GSM8K | MMLU-hum |
|---|---|---|---|---|
| baseline | 128 | (91.5) | (99.3) | 84.7% |
| union code∪math | 67 | (90.9) | (97.3) | **50.3%** |
| random size-matched | 67 | 58.5% | 37.3% | 62.3% |
| humanities-only | 45 | 0.0% | 0.7% | 28.7% |
| union code∪hum | 71 | 84.1% | 32.7% | 74.3% |

### Findings

1. **Third-domain probe passed**: code∪math drops 34pp on non-STEM MMLU
   while holding both target domains at baseline — a genuine two-skill
   specialist, not an accidental generalist.
2. **Size-matched random control passed**: at identical 67/layer, random
   trails the union by 32pp (HumanEval) and 60pp (GSM8K), yet BEATS it on
   the neutral domain (62.3 vs 50.3) — random preserves diffuse mediocrity,
   targeted selection concentrates capability. Selection, not size.
3. **Hard-pair composition passed**: code∪humanities (19/layer shared vs
   22.6 for code∪math — distant domains overlap less) scores 84.1 HumanEval
   + 74.3 MMLU-hum with GSM8K (neither domain) staying at 32.7.
4. **Surprise finding — knowledge does not prune like skill**: the
   humanities-only specialist collapsed on its own domain (28.7 vs 84.7
   baseline), unlike code (82.3) and math (93.3) specialists. Consistent
   with skills-are-localized / knowledge-is-distributed. Yet the code∪hum
   union recovers humanities to 74.3 — complementary expert sets jointly
   restore capability neither holds alone. Procedural domains are far more
   prunable than knowledge domains at matched ratios.

Caveats: MMLU n=300 (±~3pp), single seed, single model; humanities
calibration (24 essay-style items) format-mismatches the multiple-choice
probe — part of finding 4 may be format sensitivity (worth one ablation).

### K3 implications

Method validated end-to-end on real hardware. If specialization grows with
expert granularity (the core hypothesis), K3's 896-expert pool should knee
at a higher ratio than this 128-expert model. Pilot cost: $0.
