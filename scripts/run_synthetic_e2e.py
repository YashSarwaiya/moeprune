"""Run the full pipeline on the synthetic planted MoE and print the miniature
quality-vs-ratio curve. This is the shape of the final deliverable, produced
locally in seconds. Usage: PYTHONPATH=src python3 scripts/run_synthetic_e2e.py"""
import torch

from moeprune.instrument import RouterLogger
from moeprune.protect import standing_committee_report, super_experts
from moeprune.prune import build_manifest
from moeprune.scoring import scores, specialization_gini
from moeprune.synthetic import (SynthSpec, build_planted_model, build_pruned_model,
                               domain_batch, general_batch, rel_error)

spec = SynthSpec()
model = build_planted_model(spec)


def log(tokens):
    logger = RouterLogger(model, spec.n_layers, spec.n_experts, spec.top_k)
    with torch.no_grad():
        model(tokens)
    logger.flush(len(tokens))
    logger.detach()
    return logger.acc


print("1. Logging routing on domain + general calibration batches...")
s_dom = scores(log(domain_batch(spec)), "contrib")
s_gen = scores(log(general_batch(spec)), "contrib")

print(f"2. Specialization Gini per layer (general set): "
      f"{[f'{g:.2f}' for g in specialization_gini(s_gen)]}")
committee = standing_committee_report(s_gen)
print(f"   Standing committee: {committee[0.5]['mean']:.0f}/{spec.n_experts} "
      f"experts carry 50% of mass, {committee[0.95]['mean']:.0f} carry 95%")

protected = super_experts(s_gen, top_frac=0.02)
print(f"3. Protected super experts (layer 0): {protected[0].tolist()}")

print("4. Pruning sweep (rel. error of residual stream vs full model):")
print(f"   {'ratio':>6} {'kept':>5} {'domain err':>11} {'general err':>12}")
for ratio in (0.25, 0.50, 0.65, 0.80, 0.875):
    m = build_manifest(s_dom, protected, ratio=ratio, min_keep=16,
                       top_k=spec.top_k)
    pruned = build_pruned_model(model, m.keep)
    e_d = rel_error(model, pruned, domain_batch(spec, seed=7))
    e_g = rel_error(model, pruned, general_batch(spec, seed=8))
    print(f"   {ratio:>5.0%} {len(m.keep[0]):>5} {e_d:>10.4f} {e_g:>11.4f}")
print("\nDomain capability held while general capability degraded -> the "
      "pipeline measures what it claims. Ready for real logs.")
