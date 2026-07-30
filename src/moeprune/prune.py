"""Pruning manifests and checkpoint surgery.

A manifest is the complete, auditable record of one sweep point: which experts
survive in each layer and why. Checkpoint surgery is a pure key-remapping
problem — remove pruned experts' FFN tensors, slice the router's rows down to
survivors, densely renumber. It runs on a CPU box; no GPU involved.

Tensor naming differs per checkpoint, so patterns live in KeyMap. Defaults
match the synthetic model; the real K3 KeyMap is filled in after one look at
the shard index (scripts/inspect_checkpoint.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Manifest:
    ratio: float
    method: str
    keep: list[list[int]]           # per layer, sorted surviving expert ids
    protected: list[list[int]]      # per layer, ids kept due to protection
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, default=int)  # tolerate stray np ints

    @staticmethod
    def load(path: str) -> "Manifest":
        with open(path) as f:
            return Manifest(**json.load(f))


def build_manifest(domain_score: np.ndarray, protected: list[np.ndarray],
                   ratio: float, min_keep: int, top_k: int,
                   method: str = "contrib") -> Manifest:
    """Keep the top-scored experts per layer at the target ratio, always
    including the protected set. min_keep guards routing slack (>= 2*top_k
    enforced)."""
    n_layers, n_experts = domain_score.shape
    min_keep = max(min_keep, 2 * top_k)
    n_keep = max(min_keep, int(round(n_experts * (1.0 - ratio))))
    keep, prot = [], []
    for l in range(n_layers):
        p = np.asarray(protected[l], dtype=int)
        ranked = np.argsort(-domain_score[l], kind="stable")
        chosen = [int(x) for x in p]     # plain ints: keep must be JSON-clean
        seen = set(chosen)
        for e in ranked:
            if len(chosen) >= max(n_keep, len(p)):
                break
            if int(e) not in seen:
                chosen.append(int(e))
                seen.add(int(e))
        keep.append(sorted(chosen))
        prot.append(sorted(int(x) for x in p))
    return Manifest(ratio=ratio, method=method, keep=keep, protected=prot,
                    meta={"n_experts": n_experts, "n_keep_target": n_keep})


@dataclass
class KeyMap:
    """Regexes over state-dict keys. expert_pattern groups: (layer, expert).
    gate_patterns group: (layer,); those tensors have one row per expert and
    are sliced to survivors (router weight, and e-score bias if present)."""
    expert_pattern: str = r"^(.*layers\.(\d+)\.moe\.experts\.)(\d+)(\..+)$"
    gate_patterns: tuple[str, ...] = (r"^.*layers\.(\d+)\.moe\.gate\.(weight|bias)$",)


def apply_to_state_dict(sd: dict, manifest: Manifest, keymap: KeyMap) -> dict:
    """Return a new state dict with pruned experts removed and renumbered.
    Works shard-by-shard for the real checkpoint (keys not in any pattern pass
    through untouched)."""
    expert_re = re.compile(keymap.expert_pattern)
    gate_res = [re.compile(p) for p in keymap.gate_patterns]
    remap = [{old: new for new, old in enumerate(layer_keep)}
             for layer_keep in manifest.keep]
    out = {}
    for key, tensor in sd.items():
        m = expert_re.match(key)
        if m:
            layer, expert = int(m.group(2)), int(m.group(3))
            if expert in remap[layer]:
                out[f"{m.group(1)}{remap[layer][expert]}{m.group(4)}"] = tensor
            continue  # pruned expert: dropped
        for g in gate_res:
            gm = g.match(key)
            if gm:
                layer = int(gm.group(1))
                tensor = tensor[manifest.keep[layer]]
                break
        out[key] = tensor
    return out
