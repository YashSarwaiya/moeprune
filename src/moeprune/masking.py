"""Prune-by-masking: evaluate a pruning manifest without touching the weights.

A forward hook on each gate Linear sets pruned experts' logits to dtype-min,
so top-k can never select them; softmax renormalization over the survivors
then matches what a surgically pruned checkpoint computes (verified exactly in
tests/test_e2e_synthetic.py). Compute savings aren't realized — this is for
measuring the QUALITY curve while the full model sits in VRAM, which is the
deliverable. Storage cost of a sweep: zero.
"""
from __future__ import annotations

import re

import torch

from .prune import Manifest


class ExpertMasker:
    def __init__(self, model: torch.nn.Module, n_experts: int,
                 gate_pattern: str = r"layers\.(\d+)\.moe\.gate$"):
        self.n_experts = n_experts
        self._masks: dict[int, torch.Tensor] = {}  # layer -> bool, True=pruned
        self._handles = []
        gate_re = re.compile(gate_pattern)
        for name, module in model.named_modules():
            m = gate_re.search(name)
            if m:
                self._handles.append(module.register_forward_hook(
                    self._make_hook(int(m.group(1)))))
        if not self._handles:
            raise ValueError("no modules matched gate_pattern")

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            mask = self._masks.get(layer)
            if mask is None:
                return output
            return output.masked_fill(mask.to(output.device),
                                      torch.finfo(output.dtype).min)
        return hook

    def apply(self, manifest: Manifest) -> None:
        self._masks = {}
        for layer, keep in enumerate(manifest.keep):
            mask = torch.ones(self.n_experts, dtype=torch.bool)
            mask[torch.as_tensor(keep, dtype=torch.long)] = False
            self._masks[layer] = mask

    def clear(self) -> None:
        """Back to the unpruned model (baseline evals)."""
        self._masks = {}

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
