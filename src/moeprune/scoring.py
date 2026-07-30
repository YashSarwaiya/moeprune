"""Expert-contribution scoring from routing logs.

Three scores, in increasing order of fidelity:

- freq:      activation count. Conflates syntactic regularity with importance
             (Mixtral lesson) — kept only as a baseline for ablations.
- gate_mass: sum of renormalized gate weights. What the router *believes*.
- contrib:   sum of gate_weight * ||expert_output||. What the expert actually
             *adds* to the residual stream. Preferred (norm-aware, per the
             method sketch); requires the PyTorch capture path.

All scores are per (layer, expert) and only comparable within a layer, so
ranking is always per-layer.
"""
from __future__ import annotations

import numpy as np

from .instrument import ActivationAccumulator

METHODS = ("freq", "gate_mass", "contrib")


def scores(acc: ActivationAccumulator, method: str = "contrib") -> np.ndarray:
    """[n_layers, n_experts] score matrix. Falls back contrib -> gate_mass if
    the capture path had no norms (llama.cpp)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "contrib" and not acc.has_norms:
        method = "gate_mass"
    return {"freq": acc.count, "gate_mass": acc.gate_mass,
            "contrib": acc.contrib}[method].copy()


def layer_rankings(score: np.ndarray) -> np.ndarray:
    """Per-layer expert ids sorted by descending score. [n_layers, n_experts]."""
    return np.argsort(-score, axis=1, kind="stable")


def cumulative_mass_curve(score: np.ndarray, layer: int) -> np.ndarray:
    """Fraction of layer's total score captured by top-n experts, for all n.
    Answers 'how much of the 896 is standing committee' per layer."""
    s = np.sort(score[layer])[::-1]
    total = s.sum()
    return np.cumsum(s) / total if total > 0 else np.zeros_like(s)


def specialization_gini(score: np.ndarray) -> np.ndarray:
    """Per-layer Gini coefficient of the score distribution. High Gini = a few
    experts dominate (prunable long tail); low Gini = flat usage (dangerous to
    prune). This is the first number to look at when logs come back."""
    out = np.zeros(score.shape[0])
    for l in range(score.shape[0]):
        s = np.sort(score[l])
        n, total = len(s), s.sum()
        if total <= 0:
            continue
        out[l] = (2 * np.sum((np.arange(1, n + 1)) * s) / (n * total)) - (n + 1) / n
    return out
