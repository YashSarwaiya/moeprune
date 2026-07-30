"""Model facts and sweep configuration.

Everything K3-specific lives here so the rest of the pipeline is
architecture-agnostic. Values marked UNVERIFIED must be confirmed against the
actual HF config.json of moonshotai/Kimi-K3 before the paid runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str = "moonshotai/Kimi-K3"
    n_routed_experts: int = 896
    top_k: int = 16
    n_layers: int = 61              # UNVERIFIED: read from config.json
    n_shared_experts: int = 1       # UNVERIFIED: shared experts are never pruned
    total_params: float = 2.8e12
    active_params: float = 104e9
    checkpoint_bytes: float = 1.56e12
    # Fraction of checkpoint bytes held by *routed* expert FFNs. For fine-grained
    # MoE this dominates. UNVERIFIED: compute exactly from shard index at
    # download time (scripts/inspect_checkpoint.py prints it).
    routed_expert_bytes_fraction: float = 0.95


@dataclass
class SweepSpec:
    # Fraction of ROUTED experts removed, per sweep point.
    ratios: tuple[float, ...] = (0.25, 0.50, 0.65, 0.80, 0.90)
    # Never keep fewer than this many routed experts per layer. Routing needs
    # slack above top_k or the "choice" degenerates; 4x is a conservative floor.
    min_keep_per_layer: int = 64
    scoring_method: str = "contrib"  # freq | gate_mass | contrib
    # Super-expert protection thresholds (see protect.py).
    super_tau: float = 8.0
    super_top_frac: float = 0.01


@dataclass
class ProjectConfig:
    model: ModelSpec = field(default_factory=ModelSpec)
    sweep: SweepSpec = field(default_factory=SweepSpec)


def pruned_checkpoint_bytes(model: ModelSpec, ratio: float) -> float:
    """Checkpoint size after removing `ratio` of routed experts."""
    f = model.routed_expert_bytes_fraction
    return model.checkpoint_bytes * (1.0 - f * ratio)
