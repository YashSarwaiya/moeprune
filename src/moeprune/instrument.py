"""Router instrumentation.

Two capture paths, one output format:

1. PyTorch forward hooks (this module) — for any HF-style MoE where the gate is
   an nn.Linear producing logits over experts. We hook the gate, recompute
   top-k + softmax ourselves, and optionally hook each expert module to record
   output norms (needed for the norm-aware "contrib" score). Used for vLLM
   worker processes and for the synthetic validation model.

2. llama.cpp routing dump (parse_llamacpp_log) — for the cheap CPU-box
   calibration path. Requires a ~30-line patch to the Unsloth fork emitting one
   JSONL record per (layer, token): {"l": int, "e": [ids], "w": [weights]}.
   The patch point is the expert-selection site in build_moe_ffn (ggml graph
   callback on the top-k/softmax nodes); no norms available on this path, so
   scoring falls back to gate_mass.

Output: an ActivationAccumulator — dense [n_layers, n_experts] float64 arrays
(count, gate_mass, contrib) saved as .npz. Logs never store per-token records
for the real model; 60k calibration tokens x 61 layers x 16 experts is fine,
but aggregation-on-the-fly keeps the GPU-side footprint trivial.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class ActivationAccumulator:
    n_layers: int
    n_experts: int
    count: np.ndarray = field(init=False)      # times expert was in top-k
    gate_mass: np.ndarray = field(init=False)  # sum of renormalized gate weights
    contrib: np.ndarray = field(init=False)    # sum of gate_weight * ||expert_out||_2
    tokens_seen: int = 0
    has_norms: bool = False

    def __post_init__(self) -> None:
        shape = (self.n_layers, self.n_experts)
        self.count = np.zeros(shape)
        self.gate_mass = np.zeros(shape)
        self.contrib = np.zeros(shape)

    def add(self, layer: int, expert_ids: np.ndarray, weights: np.ndarray,
            out_norms: np.ndarray | None = None) -> None:
        """Add one batch of routing decisions for one layer.

        expert_ids, weights: flat arrays over (token, slot) pairs.
        out_norms: matching ||f_e(x_t)||_2 values, if the capture path has them.
        """
        np.add.at(self.count[layer], expert_ids, 1.0)
        np.add.at(self.gate_mass[layer], expert_ids, weights)
        if out_norms is not None:
            np.add.at(self.contrib[layer], expert_ids, weights * out_norms)
            self.has_norms = True

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, count=self.count, gate_mass=self.gate_mass, contrib=self.contrib,
            tokens_seen=self.tokens_seen, has_norms=self.has_norms)

    @staticmethod
    def load(path: str) -> "ActivationAccumulator":
        z = np.load(path)
        acc = ActivationAccumulator(*z["count"].shape)
        acc.count, acc.gate_mass, acc.contrib = z["count"], z["gate_mass"], z["contrib"]
        acc.tokens_seen = int(z["tokens_seen"])
        acc.has_norms = bool(z["has_norms"])
        return acc


class RouterLogger:
    """Attach forward hooks to a PyTorch MoE model and aggregate routing stats.

    gate_pattern must match the gate Linear's module name with one group: layer
    index. expert_pattern must match each expert module with two groups: layer
    index, expert index. Defaults match `layers.{L}.moe.gate` /
    `layers.{L}.moe.experts.{E}` (synthetic model); real K3 patterns are set
    after inspecting the loaded module tree.

    Norm capture assumes the implementation dispatches each expert on its
    routed tokens in ascending token order (true for Mixtral-style loops).
    Verify once on the real model with verify_norm_alignment().
    """

    def __init__(self, model: torch.nn.Module, n_layers: int, n_experts: int,
                 top_k: int,
                 gate_pattern: str = r"layers\.(\d+)\.moe\.gate$",
                 expert_pattern: str = r"layers\.(\d+)\.moe\.experts\.(\d+)$",
                 capture_norms: bool = True):
        self.acc = ActivationAccumulator(n_layers, n_experts)
        self.top_k = top_k
        self._handles = []
        # Per-forward staging: layer -> (ids [T,k], weights [T,k])
        self._routing: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # layer -> expert -> norms of that expert's outputs this forward
        self._norms: dict[int, dict[int, np.ndarray]] = {}

        gate_re, expert_re = re.compile(gate_pattern), re.compile(expert_pattern)
        for name, module in model.named_modules():
            m = gate_re.search(name)
            if m:
                layer = int(m.group(1))
                self._handles.append(module.register_forward_hook(
                    self._make_gate_hook(layer)))
                continue
            if capture_norms:
                m = expert_re.search(name)
                if m:
                    layer, expert = int(m.group(1)), int(m.group(2))
                    self._handles.append(module.register_forward_hook(
                        self._make_expert_hook(layer, expert)))
        if not self._handles:
            raise ValueError("no modules matched gate_pattern; check module names")

    def _make_gate_hook(self, layer: int):
        def hook(_module, _inputs, output):
            # Robust to gate variants: plain logits tensor (classic), a tuple
            # containing a [.., n_experts] tensor (router-class refactors), or
            # already-softmaxed probabilities (detected, softmax skipped).
            t = output
            if isinstance(t, (tuple, list)):
                cands = [x for x in t if torch.is_tensor(x) and x.ndim >= 2
                         and x.shape[-1] == self.acc.n_experts]
                if not cands:
                    raise RuntimeError(
                        f"gate hook (layer {layer}): no [.., {self.acc.n_experts}]"
                        f" tensor in gate output tuple")
                t = cands[0]
            t = t.detach().float().reshape(-1, t.shape[-1])
            row_sums = t.sum(dim=-1)
            if float(t.min()) >= 0 and torch.allclose(
                    row_sums, torch.ones_like(row_sums), atol=1e-3):
                weights = t                      # already probabilities
            else:
                weights = torch.softmax(t, dim=-1)
            topw, topi = torch.topk(weights, self.top_k, dim=-1)
            topw = topw / topw.sum(dim=-1, keepdim=True)  # renorm over selected
            self._routing[layer] = (topi.cpu().numpy(), topw.cpu().numpy())
        return hook

    def _make_expert_hook(self, layer: int, expert: int):
        def hook(_module, _inputs, output):
            norms = output.detach().float().reshape(-1, output.shape[-1]).norm(dim=-1)
            self._norms.setdefault(layer, {})[expert] = norms.cpu().numpy()
        return hook

    def flush(self, n_tokens: int) -> None:
        """Call after each forward pass to fold staged data into the accumulator."""
        for layer, (ids, weights) in self._routing.items():
            flat_ids, flat_w = ids.reshape(-1), weights.reshape(-1)
            out_norms = None
            layer_norms = self._norms.get(layer)
            if layer_norms is not None:
                # Rebuild per-(token,slot) norms from per-expert output stacks.
                # Each expert processes its routed tokens in ascending token
                # order and top-k ids are unique within a token, so walking
                # tokens in order consumes each expert's norm stack in sync.
                out_norms = np.zeros_like(flat_w)
                cursor = {e: 0 for e in layer_norms}
                for t in range(ids.shape[0]):
                    for s in range(ids.shape[1]):
                        e = int(ids[t, s])
                        if e in layer_norms and cursor[e] < len(layer_norms[e]):
                            out_norms[t * ids.shape[1] + s] = layer_norms[e][cursor[e]]
                            cursor[e] += 1
            self.acc.add(layer, flat_ids, flat_w, out_norms)
        self.acc.tokens_seen += n_tokens
        self._routing.clear()
        self._norms.clear()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()


def parse_llamacpp_log(path: str, n_layers: int, n_experts: int) -> ActivationAccumulator:
    """Fold a llama.cpp routing-dump JSONL into an accumulator (no norms)."""
    acc = ActivationAccumulator(n_layers, n_experts)
    tokens = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            ids = np.asarray(rec["e"], dtype=np.int64)
            w = np.asarray(rec["w"], dtype=np.float64)
            acc.add(rec["l"], ids, w / max(w.sum(), 1e-12))
            tokens.add((rec.get("t", len(tokens))))
    acc.tokens_seen = len(tokens)
    return acc
