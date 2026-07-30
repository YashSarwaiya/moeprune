"""Tiny synthetic MoE with *planted* structure, for validating the pipeline.

Ground truth we control:
  - domain tokens (ids < n_domain_tokens) share a common embedding direction u
  - "domain experts" have gate rows aligned with u -> they win routing on
    domain tokens
  - "super experts" have a large gate bias -> they win routing on everything

If the pipeline is correct, scoring on a domain batch must surface
domain+super experts, protection on a general batch must flag the supers, and
a pruned model must track the full model closely on domain tokens while
degrading on general tokens. None of this validates the science on K3 — it
validates that the code measures what it claims to measure.

Module names deliberately match RouterLogger's default patterns
(layers.{L}.moe.gate / layers.{L}.moe.experts.{E}) so the exact hook path used
on the real model is exercised.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SynthSpec:
    vocab: int = 256
    n_domain_tokens: int = 64
    d: int = 32
    n_layers: int = 4
    n_experts: int = 64
    top_k: int = 8
    super_ids: tuple[int, ...] = (0, 1)
    domain_ids: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    seed: int = 0


class MoELayer(nn.Module):
    def __init__(self, d: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d, n_experts, bias=True)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, d]. Mixtral-style dispatch: each expert sees its routed tokens
        # in ascending token order (RouterLogger's norm alignment relies on this).
        logits = self.gate(x)
        weights = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(weights, self.top_k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topi == e)
            if len(token_idx) == 0:
                continue
            out[token_idx] += topw[token_idx, slot_idx, None] * expert(x[token_idx])
        return out


class Block(nn.Module):
    def __init__(self, d: int, n_experts: int, top_k: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mix = nn.Linear(d, d)  # stand-in for attention: mixes the stream
        self.moe = MoELayer(d, n_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        return x + self.mix(h) + self.moe(h)


class TinyMoE(nn.Module):
    def __init__(self, spec: SynthSpec, n_experts_per_layer: list[int] | None = None):
        super().__init__()
        self.spec = spec
        counts = n_experts_per_layer or [spec.n_experts] * spec.n_layers
        self.emb = nn.Embedding(spec.vocab, spec.d)
        self.layers = nn.ModuleList(
            Block(spec.d, counts[i], spec.top_k) for i in range(spec.n_layers))
        self.ln_f = nn.LayerNorm(spec.d)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ln_f(self.hidden(tokens))

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pre-final-norm residual stream — what retention metrics compare
        (ln_f flattens magnitude differences and masks pruning damage)."""
        x = self.emb(tokens)
        for layer in self.layers:
            x = layer(x)
        return x


def build_planted_model(spec: SynthSpec) -> TinyMoE:
    torch.manual_seed(spec.seed)
    model = TinyMoE(spec)
    with torch.no_grad():
        u = torch.randn(spec.d)
        u = u / u.norm()
        # Domain tokens share direction u (plus mild noise).
        for t in range(spec.n_domain_tokens):
            noise = torch.randn(spec.d) * 0.2
            model.emb.weight[t] = 2.0 * u + noise
        for block in model.layers:
            # Shrink the attention stand-in so the MoE output dominates the
            # stream — otherwise retention metrics can't see pruning at all.
            block.mix.weight *= 0.3
            gate = block.moe.gate
            gate.bias.zero_()
            for e in spec.super_ids:
                gate.bias[e] = 2.0   # reliably in top-k everywhere, but not
                # so dominant that general tokens survive on supers alone
            for e in spec.domain_ids:
                gate.weight[e] = 2.0 * u + 0.1 * torch.randn(spec.d)
    model.eval()
    return model


def build_pruned_model(full: TinyMoE, keep: list[list[int]]) -> TinyMoE:
    """Construct the pruned model via the real checkpoint-surgery path
    (prune.apply_to_state_dict), not by copying modules — so the test covers
    the same code that will rewrite K3 shards."""
    from .prune import KeyMap, Manifest, apply_to_state_dict

    spec = full.spec
    manifest = Manifest(ratio=0.0, method="test", keep=keep,
                        protected=[[] for _ in keep])
    new_sd = apply_to_state_dict(full.state_dict(), manifest, KeyMap())
    pruned = TinyMoE(spec, n_experts_per_layer=[len(k) for k in keep])
    pruned.load_state_dict(new_sd)
    pruned.eval()
    return pruned


def domain_batch(spec: SynthSpec, n: int = 512, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, spec.n_domain_tokens, (n,), generator=g)


def general_batch(spec: SynthSpec, n: int = 512, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(spec.n_domain_tokens, spec.vocab, (n,), generator=g)


def retention(full: TinyMoE, pruned: TinyMoE, tokens: torch.Tensor) -> float:
    """Mean cosine similarity of final hidden states, full vs pruned."""
    with torch.no_grad():
        a, b = full(tokens), pruned(tokens)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def rel_error(full: TinyMoE, pruned: TinyMoE, tokens: torch.Tensor) -> float:
    """Mean relative L2 error of the pre-norm residual stream, full vs pruned.
    The sensitive metric: ~0 where routing survived pruning, large where the
    token's experts were deleted."""
    with torch.no_grad():
        a, b = full.hidden(tokens), pruned.hidden(tokens)
    return float((torch.norm(a - b, dim=-1) / torch.norm(a, dim=-1)).mean())
