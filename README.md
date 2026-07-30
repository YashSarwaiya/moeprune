# moeprune — training-free domain pruning and recombination of MoE experts

Code and data for **"Carve, Then Compose: Training-Free Domain Pruning and
Recombination of Experts in Mixture-of-Experts LLMs."**

**Headline:** on Qwen3-30B-A3B (128 experts/layer), removing 65% of routed
experts with a coding-domain score retains 82.3% HumanEval pass@1 (baseline
91.5%) while GSM8K collapses to 33.3% — and independently pruned coding and
math expert sets can be **unioned post hoc** into a 67-expert two-skill model
scoring 90.9 / 97.3, within 1–2 points of the full model. No retraining, one
GPU, ~30 calibration examples per domain.

## Results at a glance

| Variant | Kept/layer | HumanEval | GSM8K | MMLU-hum |
|---|---|---|---|---|
| baseline | 128 | 91.5 | 99.3 | 84.7 |
| coding-pruned 50% | 64 | 89.0 | 94.0 | — |
| coding-pruned 65% | 45 | 82.3 | 33.3 | — |
| **random @ 65%** | 45 | **0.6** | 16.0 | — |
| math-pruned 65% | 45 | 0.0 | 93.3 | — |
| **code ∪ math** | 67 | **90.9** | **97.3** | 50.3 |
| random size-matched | 67 | 58.5 | 37.3 | 62.3 |
| **code ∪ humanities** | 71 | **84.1** | 32.7 | **74.3** |

Full tables, caveats, and the surprising skills-vs-knowledge asymmetry are in
[RESULTS.md](RESULTS.md); the paper source is in [paper/](paper/).

## Quickstart

Validate the whole pipeline locally in seconds, no GPU, no model download —
a synthetic MoE with *planted* domain and super experts checks that the
scoring, protection, masking, and surgery code find what they claim to:

```bash
pip install -e .
PYTHONPATH=src python3 tests/test_e2e_synthetic.py   # 7 tests
PYTHONPATH=src python3 scripts/run_synthetic_e2e.py  # miniature curve
```

## Reproducing the paper

One GPU with ~80 GB (we used a single B200), ~10 GPU-hours total:

```bash
bash scripts/pilot_qwen/predownload.sh    # model + eval sets (run once)
bash scripts/pilot_qwen/fetch_mmlu.sh     # non-STEM knowledge probe

python scripts/pilot_qwen/pilot.py   --out results/pilot     # §5.1 sweep
python scripts/pilot_qwen/pilot.py   --skip-baseline \
    --ratios 0.70,0.75 --random-ratios 0.50,0.65,0.80 \
    --out results/controls                                   # §5.2 controls
python scripts/pilot_qwen/union.py   --out results/union     # §5.3 composition
python scripts/pilot_qwen/skeptic.py --out results/skeptic   # §5.2–5.4 probes
```

SLURM users: `*.sbatch` wrappers for each are in `scripts/pilot_qwen/`
(edit the account/partition headers for your cluster).

Requires `transformers>=4.51,<4.56` — v5 restructured MoE internals (the gate
returns a tuple, fused experts), which breaks the router hooks. The gate/expert
module patterns are regex-configurable for other architectures.

## How it works

1. **Instrument** ([instrument.py](src/moeprune/instrument.py)) — hooks capture
   per-token top-k expert ids, renormalized gate weights, and expert output
   norms during a teacher-forced prefill. No generation needed.
2. **Score** ([scoring.py](src/moeprune/scoring.py)) — rank experts per layer by
   `Σ gate_weight · ‖expert_output‖₂`, the contribution actually added to the
   residual stream. Raw routing frequency is available as a baseline but
   conflates syntax with importance.
3. **Protect** ([protect.py](src/moeprune/protect.py)) — flag "super experts" on
   a *general* calibration set (>8× layer median mass, or top 2%) and keep them
   in every manifest, whatever the domain.
4. **Prune** ([prune.py](src/moeprune/prune.py)) — emit a manifest of survivors
   per layer; apply it either by real checkpoint surgery (drop expert tensors,
   slice router rows, renumber) or by
   **[masking](src/moeprune/masking.py)** — router logits set to −∞, exactly
   output-equivalent (verified <1e-5) and free of storage cost.
5. **Compose** — per-layer set union of two manifests. Nothing is recomputed.

## Repo map

```
src/moeprune/       pipeline: instrument, scoring, protect, prune, masking, costs
calibration/       coding (30), math (28), humanities (24), general (16) items
scripts/           synthetic demo, cost model, checkpoint tools
scripts/pilot_qwen/  the four experiments from the paper + SLURM wrappers
tests/             end-to-end validation on a planted-structure synthetic MoE
paper/             LaTeX source and markdown draft
RESULTS.md         every measured number, with caveats
```

## Status and roadmap

The Qwen3-30B study is complete. The pipeline is architecture-agnostic by
design (module patterns are configurable), and the next target is **Kimi K3**
— 896 experts/layer, the most fine-grained public MoE — to test whether the
knee moves right as expert granularity increases. See the "Future Work"
section of the paper.

Contributions, replications on other MoE models, and negative results are all
welcome — open an issue.

## Citation

```bibtex
@article{sarwaiya2026carve,
  title  = {Carve, Then Compose: Training-Free Domain Pruning and
            Recombination of Experts in Mixture-of-Experts LLMs},
  author = {Sarwaiya, Yash},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
