# Carve, Then Compose: Training-Free Domain Pruning and Recombination of Experts in Mixture-of-Experts LLMs

**Yash Sarwaiya**
Independent Researcher

*Working draft. Submission version: `paper/main.tex`. Code and results: https://github.com/YashSarwaiya/moeprune*

## Abstract

Expert pruning of Mixture-of-Experts (MoE) language models is believed to be
safe only up to roughly 25% without retraining. We show that this ceiling is
an artifact of demanding *general* capability, and that when only one domain
must survive, an off-the-shelf fine-grained MoE tolerates far deeper cuts.
On Qwen3-30B-A3B (128 experts/layer, top-8), a norm-aware contribution score
computed from a teacher-forced prefill over only ~30 domain demonstrations
retains 97% of baseline HumanEval pass@1 with 50% of routed experts removed,
and 90% with 65% removed — while GSM8K collapses from 99.3% to 33.3%,
confirming genuinely domain-targeted compression. A size-matched random
control scores 26.8% HumanEval at the 50% ratio, an advantage of 62 points
for informed selection. We further show that domain-pruned variants
*compose*: the per-layer union of independently derived coding and math
expert sets (52% of experts) scores within 1–2 points of the full model on
both HumanEval (90.9%) and GSM8K (97.3%), while dropping 34 points on a
non-STEM MMLU probe — a true two-skill specialist assembled post hoc,
with no retraining. Composition holds for distant domain pairs
(code∪humanities: 84.1% HumanEval, 74.3% MMLU-humanities) whose keep-sets
overlap less than sibling pairs. Finally, we observe a sharp asymmetry:
procedural domains (coding, math) prune cleanly, but a knowledge domain
(humanities) does not — the humanities-only specialist collapses even on its
own domain (28.7% vs 84.7% baseline), yet unions restore it. Our results
suggest fine-grained MoE models are post-hoc composable at the expert level,
and motivate applying the recipe to the most fine-grained public MoE, Kimi
K3 (896 experts/layer).

## 1 Introduction

Sparse Mixture-of-Experts architectures dominate the largest open language
models, but their checkpoints are monolithic: a user who needs only coding
ability must nevertheless store and serve every expert, including those
that exist to support poetry, law, and the long tail of world knowledge.
One-shot expert pruning offers relief, but published results converge on a
ceiling near 25% of experts before general performance degrades
[HEAPr; MoNE; MoE-I2], with ~50% reachable only with retraining or
distillation [MoE-Pruner; SlimQwen].

We ask a different question: **how much of the expert pool is needed if the
user cares about a single domain?** This reframing removes the constraint
that produced the 25% ceiling — capability outside the target domain is
deliberately sacrificed rather than preserved. Building on few-shot expert
localization [EASY-EP], we prune an off-the-shelf fine-grained MoE with a
simple, training-free pipeline and measure the full quality-versus-ratio
curve, with the controls that the one-shot pruning literature typically
omits.

Our contributions:

1. **A domain-restricted pruning curve far above the general ceiling.**
   With a norm-aware contribution score over ~30 demonstrations plus
   protection of "super experts" identified on a general set, coding
   capability survives 50% expert removal at 89.0% HumanEval pass@1
   (baseline 91.5%) and 65% removal at 82.3%. The curve holds a plateau to
   its knee at ~65%, decays steadily to 80% (45.1%), and collapses by
   87.5%.
2. **Matched random controls quantifying the value of selection.** At
   identical keep-set sizes, random expert selection scores 26.8% vs 89.0%
   (50% ratio) and 0.6% vs 82.3% (65% ratio). Informed selection — not
   redundancy — carries the result.
3. **Training-free composition of domain specialists.** The per-layer union
   of independently derived coding and math keep-sets (67/128 experts,
   75% of the sum of the parts) scores 90.9% HumanEval and 97.3% GSM8K —
   above both specialists and within 2 points of the unpruned model —
   while a non-STEM MMLU probe confirms it remains a specialist (50.3% vs
   84.7% baseline). A distant pair (code∪humanities) also composes.
4. **A skills/knowledge asymmetry.** Identical machinery that yields strong
   coding and math specialists fails to produce a humanities specialist
   (28.7% on MMLU-humanities vs 84.7% baseline), consistent with reports
   that semantic knowledge is distributed while procedural skill is
   localized [What Gets Activated; Illusion of Specialization]. Unions
   partially restore the lost capability (74.3%), indicating the deficit is
   complementary rather than absolute.

All experiments are one-shot (no gradient updates), run on a single NVIDIA
B200, and use masking that is exactly output-equivalent to checkpoint
surgery (§3.4), making the study reproducible at minimal cost.

## 2 Related Work

**General expert pruning.** HEAPr [arXiv:2509.22299] and MoNE
[arXiv:2507.00390] achieve near-lossless one-shot pruning at 20–25% on
DeepSeek/Qwen MoE families; MoNE reports ~24–25% as the maximum within 1%
loss for Qwen3-30B-A3B — the same base model we use, providing a direct
published anchor for our comparison. MoE-I² [arXiv:2411.01016] and
MoE-Pruner [arXiv:2410.12013] push to ~50% only with intra-expert
decomposition, fine-tuning, or expert-wise distillation. Scoring criteria
for one-shot pruning are surveyed in [arXiv:2606.15716]; AIMER
[arXiv:2603.18492] removes calibration dependence for the task-agnostic
setting. Our setting differs: we do not preserve general capability.

**Domain-specific pruning.** EASY-EP [arXiv:2504.06792] introduced few-shot
expert localization with output-aware importance, which our contribution
score follows in spirit. C-PRUNE [PMLR v317] clusters experts for
task-specific compression in the biomedical domain, and recent work reports
coding-domain pruning at ~50% on earlier-generation MoEs ["Half the
Experts"]. Relative to this line, we contribute (i) the full
quality-vs-ratio curve with its knee, (ii) matched random controls, and
(iii) the composition results, none of which appear in prior domain-pruning
studies to our knowledge.

**Expert merging and modularity.** An alternative to deletion is merging
similar experts [HC-SMoE arXiv:2410.08589; REAP arXiv:2510.13999, which
finds pruning superior for one-shot compression]. EMO [arXiv:2605.06663]
pretrains MoEs so that task-specific expert subsets can be selected at
deployment, arguing that standard MoEs specialize syntactically rather than
semantically. Our results provide a counterpoint: a standard,
conventionally trained MoE already supports post-hoc subset selection and
recombination for procedural domains, with no pretraining modification —
though EMO-style training may be what closes the gap we observe on
knowledge domains. Analyses of expert specialization
[arXiv:2601.10159; arXiv:2601.03425; arXiv:2604.05267] motivate our
protection of disproportionately active "super experts."

## 3 Method

Given a routed-MoE checkpoint with E experts per layer and top-k routing,
the pipeline has four stages, all training-free.

### 3.1 Routing capture

We run a teacher-forced prefill (no generation) over a small calibration
set: each item is a prompt concatenated with a reference solution. Hooks on
each layer's router record, per token, the top-k expert indices and their
renormalized gate weights, and per expert the L2 norm of its output. Only
dense per-(layer, expert) accumulators are stored.

### 3.2 Norm-aware contribution scoring

Routing frequency conflates syntactic regularity with importance, so we
score expert e in layer l as the accumulated gate-weighted output norm:

  score(l, e) = Σ_tokens g_{l,e}(x_t) · ‖f_{l,e}(x_t)‖₂

over the domain calibration set — the contribution the expert actually adds
to the residual stream, following output-aware importance [EASY-EP].

### 3.3 Super-expert protection

A small set of experts does disproportionate work regardless of domain
[arXiv:2601.03425]. On a *general* calibration set we flag experts whose
gate mass exceeds 8× the layer median, or that fall in the top 2% of a
layer's mass, and include them in every keep-set (6.5 experts/layer for our
model). Domain pruning must not delete generic infrastructure.

### 3.4 Pruning, masking, and composition

A *manifest* lists the surviving experts per layer: the protected set plus
the top domain-scored experts up to the target ratio, with a floor of 2k
experts per layer. For evaluation we *mask* rather than rewrite: a hook
sets pruned experts' router logits to −∞, which is exactly
output-equivalent to deleting the expert tensors and slicing the router
(verified to <1e−5 relative error on a synthetic model; masking makes a
sweep cost zero storage). Deployment uses real checkpoint surgery with
identical outputs.

**Composition** is the per-layer set union of two manifests. No scores are
recomputed; no calibration is rerun. The union inherits both protected
sets.

## 4 Experimental Setup

**Model.** Qwen3-30B-A3B-Instruct-2507: 48 layers, 128 routed experts per
layer, top-8 routing (norm-renormalized), bf16, evaluated on one NVIDIA
B200 with greedy decoding throughout.

**Calibration sets** (authored for this study; no benchmark items): coding
— 30 prompt+solution items across 8 languages (~15k tokens teacher-forced);
math — 28 worked word/algebra problems, deliberately disjoint from GSM8K;
humanities — 24 essay-style history/philosophy/law/arts answers; general —
16 mixed items (used only for super-expert protection).

**Benchmarks.** In-domain coding: HumanEval pass@1 (164 problems,
code-fence extraction, subprocess execution). Math: GSM8K exact-match on a
fixed 150-problem subset. Knowledge probe: 300 questions sampled from six
non-STEM MMLU subjects (philosophy, world religions, jurisprudence,
European history, marketing, prehistory). Baselines: 91.5% HumanEval,
99.3% GSM8K, 84.7% MMLU-hum. With these sample sizes, differences under
~2–3 points are within noise; we interpret only larger gaps.

**Ratios.** {25, 50, 65, 70, 75, 80, 87.5}% of routed experts removed;
87.5% coincides with the 2k-per-layer floor (16 kept, top-8 active).

## 5 Results

### 5.1 Domain-restricted pruning beats the general ceiling

| Experts removed | Kept/layer | HumanEval | GSM8K |
|---|---|---|---|
| 0% | 128 | 91.5 | 99.3 |
| 25% | 96 | 91.5 | 97.3 |
| 50% | 64 | 89.0 | 94.0 |
| 65% | 45 | 82.3 | 33.3 |
| 70% | 38 | 70.7 | 6.0 |
| 75% | 32 | 63.4 | 2.0 |
| 80% | 26 | 45.1 | 0.0 |
| 87.5% | 16 | 0.0 | 0.0 |

Coding-targeted pruning holds HumanEval within 2.5 points at 50% removal —
double the published general-pruning ceiling for this model [MoNE] — and
within 10 points at 65%, where out-of-domain GSM8K has already collapsed
(99.3 → 33.3). The in-domain curve is not smooth to the end: a plateau to
~65% (the knee), steady decay through 80%, then collapse. The 87.5% point
sits at the routing-slack floor (16 kept, 8 active) and likely reflects
that floor as much as the selection method.

Routing statistics foreshadow the curve: on the general set, mean Gini of
per-layer expert mass is 0.63, ~19/128 experts carry 50% of layer mass, and
~65/128 carry 95% — half the pool shares the last 5% of routing mass.

### 5.2 Selection, not redundancy: matched random controls

| Keep policy | Kept/layer | HumanEval | GSM8K | MMLU-hum |
|---|---|---|---|---|
| domain-scored (50%) | 64 | 89.0 | 94.0 | — |
| random (50%) | 64 | 26.8 | 40.0 | — |
| domain-scored (65%) | 45 | 82.3 | 33.3 | — |
| random (65%) | 45 | 0.6 | 16.0 | — |
| union code∪math | 67 | 90.9 | 97.3 | 50.3 |
| random size-matched | 67 | 58.5 | 37.3 | 62.3 |

Random selection at matched sizes loses 62 points of HumanEval at the 50%
ratio and 82 points at 65%. At 67/layer the pattern inverts on the neutral
domain: random *beats* the union on MMLU-hum (62.3 vs 50.3) while trailing
by 32–60 points in-domain — random preserves diffuse mediocrity, informed
selection concentrates capability into the chosen domains. (Random keep-sets
here do not receive super-expert protection; isolating protection vs
domain-scoring is left as an ablation.)

### 5.3 Domain specialists compose post hoc

| Variant | Kept/layer | HumanEval | GSM8K | MMLU-hum |
|---|---|---|---|---|
| code-only (65%) | 45 | 82.3 | 33.3 | — |
| math-only (65%) | 45 | 0.0 | 93.3 | — |
| **code∪math** | 67 | **90.9** | **97.3** | 50.3 |
| hum-only (65%) | 45 | 0.0 | 0.7 | 28.7 |
| **code∪hum** | 71 | **84.1** | 32.7 | **74.3** |
| baseline | 128 | 91.5 | 99.3 | 84.7 |

The code and math keep-sets share 22.6 experts/layer, so their union is 67
rather than 90 (75% of the sum of parts). The union exceeds *both*
specialists on their own domains and lands within 1–2 points of the
unpruned model on both benchmarks at 52% of the experts — while its 34-point
deficit on the non-STEM probe confirms it remains a two-domain specialist
rather than a recovered generalist. Composition is not limited to sibling
domains: code and humanities keep-sets overlap less (19/layer), and their
union still scores 84.1 HumanEval and 74.3 MMLU-hum while GSM8K — covered
by neither set — stays at specialist level (32.7).

The specialists themselves are strikingly disjoint in capability despite
sharing half their experts: the math-only model solves zero HumanEval
problems; the code-only model keeps residual GSM8K (33.3) plausibly via
shared arithmetic-adjacent machinery.

### 5.4 Skills prune; knowledge does not

The same pipeline that produces strong coding (82.3) and math (93.3)
specialists at the 65% ratio *fails* for humanities: the hum-only variant
scores 28.7 on its own domain, barely above the 25% choice floor. This is
consistent with analyses finding that procedural competence concentrates in
few experts while semantic knowledge is spread across many
[What Gets Activated; Illusion of Specialization]. Notably, the code∪hum
union recovers humanities to 74.3 — capability neither part exhibits alone
(code-only presumably contributes generic instruction-following and
composition machinery). Two implications: (i) domain-restricted compression
is substantially more effective for skill-like domains than knowledge-like
domains at matched ratios; (ii) deficits from aggressive pruning can be
complementary, and unions repair them cheaply.

We flag one confound: the humanities calibration set is essay-style while
the probe is multiple-choice; part of the hum-only collapse may be format
sensitivity. A format-matched ablation is future work.

## 6 Limitations

Single model, single size, single seed per point; benchmark sample sizes
give ±2–3 point noise; HumanEval and GSM8K are old benchmarks with likely
training contamination — our claims rest on *relative* retention under
identical decoding, not absolute scores. Masking realizes quality
equivalence but not the memory/latency savings, which require the
(mechanically verified) surgery path. The 87.5% point is confounded by the
routing-slack floor. Random controls omit super-expert protection. EMO
[arXiv:2605.06663] achieves related composability via modified pretraining;
we do not claim post-hoc selection matches purpose-trained modularity,
only that useful composability already exists in standard checkpoints.

## 7 Future Work

The natural target is Kimi K3 (896 experts/layer, top-16, 2.8T parameters,
released July 2026) — 7× finer expert granularity than studied here. If
specialization grows with granularity [EASY-EP], the knee should move
right, and domain-restricted compression of the strongest open model would
yield deployable specialists in the 200–400 GB range from a 1.56 TB
checkpoint. Further: knowledge-domain pruning with format-matched
calibration, per-layer adaptive ratios, expert-wise distillation to recover
the post-knee region, and N-way unions ("model à la carte").

## Acknowledgments

Computations were performed using university research computing resources.
[Before submission: replace with the exact acknowledgment wording required
by the facility's usage policy — this is a standard, generic sentence that
names no individual.]

## References

[HEAPr] Hessian-based expert pruning for MoE. arXiv:2509.22299.
[MoNE] Maximum pruning within 1% loss on Qwen MoE. arXiv:2507.00390.
[MoE-I2] Expert pruning + intra-expert decomposition. arXiv:2411.01016.
[MoE-Pruner] Pruning with expert-wise KD, Mixtral. arXiv:2410.12013.
[EASY-EP] Domain-specific pruning with few-shot demonstrations. arXiv:2504.06792.
[C-PRUNE] Cluster-driven domain-specific expert pruning. PMLR v317, 2026.
[Half the Experts] One-shot domain pruning of MoE LLMs for coding. 2026.
[What Gets Activated] Expert activation analysis. arXiv:2601.10159.
[Illusion of Specialization] Super experts and polysemy in MoE. arXiv:2601.03425.
[Do Domain Experts Exist] Domain experts across 10 MoE models. arXiv:2604.05267.
[REAP] Pruning prevails over merging for one-shot MoE compression. arXiv:2510.13999.
[HC-SMoE] Retraining-free merging via hierarchical clustering. arXiv:2410.08589.
[EMO] Pretraining MoE for emergent modularity. arXiv:2605.06663.
[AIMER] Calibration-free task-agnostic MoE pruning. arXiv:2603.18492.
[Scoring survey] Unified formulation for one-shot MoE expert scoring. arXiv:2606.15716.
[Qwen3] Qwen3 technical report. 2025.
[HumanEval] Chen et al., Evaluating LLMs trained on code. 2021.
[GSM8K] Cobbe et al., Training verifiers to solve math word problems. 2021.
[MMLU] Hendrycks et al., Measuring massive multitask language understanding. 2021.
