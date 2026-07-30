"""Turn routing logs into diagnostics + pruning manifests for the sweep.

Usage:
  python3 scripts/score_and_manifest.py --domain logs/coding.npz \
      --general logs/general.npz --outdir manifests/
"""
import argparse
import os
import sys

sys.path.insert(0, "src")

import numpy as np

from moeprune.config import ModelSpec, SweepSpec
from moeprune.instrument import ActivationAccumulator
from moeprune.protect import standing_committee_report, super_experts
from moeprune.prune import build_manifest
from moeprune.scoring import scores, specialization_gini


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="domain-set .npz log")
    ap.add_argument("--general", required=True, help="general-set .npz log")
    ap.add_argument("--outdir", default="manifests")
    args = ap.parse_args()
    model, sweep = ModelSpec(), SweepSpec()

    s_dom = scores(ActivationAccumulator.load(args.domain), sweep.scoring_method)
    s_gen = scores(ActivationAccumulator.load(args.general), sweep.scoring_method)

    gini = specialization_gini(s_gen)
    print(f"specialization Gini (general): mean {gini.mean():.3f} "
          f"min {gini.min():.3f} max {gini.max():.3f}")
    print("standing committee (general):")
    for frac, row in standing_committee_report(s_gen).items():
        print(f"  {frac:.0%} of mass carried by {row['mean']:.0f} experts/layer "
              f"(min {row['min']}, max {row['max']})")

    protected = super_experts(s_gen, tau=sweep.super_tau,
                              top_frac=sweep.super_top_frac)
    n_prot = np.mean([len(p) for p in protected])
    print(f"protected super experts: {n_prot:.1f}/layer on average")

    os.makedirs(args.outdir, exist_ok=True)
    for ratio in sweep.ratios:
        m = build_manifest(s_dom, protected, ratio=ratio,
                           min_keep=sweep.min_keep_per_layer,
                           top_k=model.top_k, method=sweep.scoring_method)
        path = os.path.join(args.outdir, f"keep_{int(ratio*100)}pct_pruned.json")
        m.save(path)
        kept = [len(k) for k in m.keep]
        print(f"ratio {ratio:.0%}: kept {min(kept)}-{max(kept)}/"
              f"{model.n_routed_experts} experts/layer -> {path}")


if __name__ == "__main__":
    main()
