"""Super-expert detection and protection sets.

Prior art (Illusion of Specialization; What Gets Activated) says a small set of
disproportionately active experts does much of the work regardless of domain.
Pruning one of those because it scored mid on a 30-prompt coding set would be
an unforced error, so protection is computed on a GENERAL calibration set and
unioned into every manifest.

An expert is 'super' in a layer if either:
  - its general-set gate mass exceeds tau * median(layer mass), or
  - it is within the top `top_frac` of experts in that layer by mass.

Shared experts (if the architecture has them) are outside the routed pool and
never touched by this pipeline at all.
"""
from __future__ import annotations

import numpy as np


def super_experts(general_score: np.ndarray, tau: float = 8.0,
                  top_frac: float = 0.01) -> list[np.ndarray]:
    """Per-layer arrays of protected expert ids, from general-domain scores."""
    n_layers, n_experts = general_score.shape
    out = []
    n_top = max(1, int(round(top_frac * n_experts)))
    for l in range(n_layers):
        s = general_score[l]
        med = np.median(s)
        by_tau = np.where(s > tau * med)[0] if med > 0 else np.array([], dtype=int)
        by_top = np.argsort(-s, kind="stable")[:n_top]
        out.append(np.union1d(by_tau, by_top))
    return out


def standing_committee_report(general_score: np.ndarray,
                              thresholds=(0.5, 0.8, 0.95)) -> dict:
    """How many experts per layer carry each fraction of total mass — a direct
    answer to the open question 'how much of the 896 can never be cut'."""
    n_layers, n_experts = general_score.shape
    report = {}
    for frac in thresholds:
        counts = []
        for l in range(n_layers):
            s = np.sort(general_score[l])[::-1]
            total = s.sum()
            if total <= 0:
                counts.append(0)
                continue
            counts.append(int(np.searchsorted(np.cumsum(s) / total, frac) + 1))
        report[frac] = {"mean": float(np.mean(counts)),
                        "min": int(np.min(counts)), "max": int(np.max(counts))}
    return report
