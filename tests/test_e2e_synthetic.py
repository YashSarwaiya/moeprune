"""End-to-end pipeline validation on the synthetic MoE with planted structure.

The chain under test is exactly the production chain:
  RouterLogger hooks -> ActivationAccumulator -> scores -> super_experts
  -> build_manifest -> apply_to_state_dict -> pruned model forward.
"""
import numpy as np
import torch

from moeprune.instrument import RouterLogger
from moeprune.protect import standing_committee_report, super_experts
from moeprune.prune import build_manifest
from moeprune.scoring import scores, specialization_gini
from moeprune.synthetic import (SynthSpec, build_planted_model, build_pruned_model,
                               domain_batch, general_batch, rel_error, retention)

SPEC = SynthSpec()


def _log(model, tokens):
    logger = RouterLogger(model, SPEC.n_layers, SPEC.n_experts, SPEC.top_k)
    with torch.no_grad():
        model(tokens)
    logger.flush(len(tokens))
    logger.detach()
    return logger.acc


def _accs():
    model = build_planted_model(SPEC)
    return (model,
            _log(model, domain_batch(SPEC)),
            _log(model, general_batch(SPEC)))


def test_instrumentation_shapes_and_norms():
    _, acc_dom, _ = _accs()
    assert acc_dom.count.shape == (SPEC.n_layers, SPEC.n_experts)
    assert acc_dom.tokens_seen == 512
    # Every token contributes top_k renormalized weights summing to 1.
    per_layer_mass = acc_dom.gate_mass.sum(axis=1)
    assert np.allclose(per_layer_mass, 512.0, rtol=1e-4)
    assert acc_dom.has_norms  # expert-output norms captured for contrib score


def test_domain_scoring_finds_planted_experts():
    _, acc_dom, _ = _accs()
    s = scores(acc_dom, "contrib")
    planted = set(SPEC.super_ids) | set(SPEC.domain_ids)
    for layer in range(SPEC.n_layers):
        top = set(np.argsort(-s[layer])[:len(planted)])
        overlap = len(top & planted) / len(planted)
        assert overlap >= 0.8, f"layer {layer}: planted-expert recall {overlap}"


def test_super_expert_protection_from_general_set():
    _, _, acc_gen = _accs()
    prot = super_experts(scores(acc_gen, "contrib"), tau=8.0, top_frac=0.02)
    for layer in range(SPEC.n_layers):
        assert set(SPEC.super_ids) <= set(prot[layer].tolist()), \
            f"layer {layer}: super experts not protected"


def test_manifest_keeps_planted_at_75pct_pruning():
    _, acc_dom, acc_gen = _accs()
    prot = super_experts(scores(acc_gen, "contrib"), top_frac=0.02)
    manifest = build_manifest(scores(acc_dom, "contrib"), prot, ratio=0.75,
                              min_keep=16, top_k=SPEC.top_k)
    planted = set(SPEC.super_ids) | set(SPEC.domain_ids)
    for layer in range(SPEC.n_layers):
        kept = set(manifest.keep[layer])
        assert len(kept) == 16
        assert planted <= kept, f"layer {layer}: planted experts pruned"
    # save/load roundtrip must survive numpy ints from the protection arrays
    import tempfile
    from moeprune.prune import Manifest
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as tmp:
        manifest.save(tmp.name)
        loaded = Manifest.load(tmp.name)
    assert loaded.keep == manifest.keep and loaded.ratio == manifest.ratio


def test_pruned_model_retains_domain_over_general():
    model, acc_dom, acc_gen = _accs()
    prot = super_experts(scores(acc_gen, "contrib"), top_frac=0.02)
    manifest = build_manifest(scores(acc_dom, "contrib"), prot, ratio=0.75,
                              min_keep=16, top_k=SPEC.top_k)
    pruned = build_pruned_model(model, manifest.keep)
    e_dom = rel_error(model, pruned, domain_batch(SPEC, seed=7))
    e_gen = rel_error(model, pruned, general_batch(SPEC, seed=8))
    r_dom = retention(model, pruned, domain_batch(SPEC, seed=7))
    # Domain capability survives 75% pruning; general capability pays for it.
    assert r_dom > 0.99, f"domain retention {r_dom:.3f}"
    assert e_dom < 0.01, f"domain rel error {e_dom:.4f}"
    assert e_gen > e_dom + 0.02, f"no tradeoff: dom {e_dom:.4f} gen {e_gen:.4f}"


def test_masking_equals_surgery():
    """A masked full model must produce the same outputs as the surgically
    pruned checkpoint — this is what licenses the zero-storage sweep."""
    from moeprune.masking import ExpertMasker
    model, acc_dom, acc_gen = _accs()
    prot = super_experts(scores(acc_gen, "contrib"), top_frac=0.02)
    manifest = build_manifest(scores(acc_dom, "contrib"), prot, ratio=0.75,
                              min_keep=16, top_k=SPEC.top_k)
    surgically_pruned = build_pruned_model(model, manifest.keep)
    masker = ExpertMasker(model, SPEC.n_experts)
    masker.apply(manifest)
    for batch in (domain_batch(SPEC, seed=11), general_batch(SPEC, seed=12)):
        with torch.no_grad():
            masked_out = model.hidden(batch)
            pruned_out = surgically_pruned.hidden(batch)
        err = (torch.norm(masked_out - pruned_out, dim=-1)
               / torch.norm(pruned_out, dim=-1)).max()
        assert err < 1e-5, f"mask/surgery divergence {err:.2e}"
    # clear() must restore the untouched model exactly
    masker.clear()
    with torch.no_grad():
        restored = model.hidden(domain_batch(SPEC, seed=11))
        original = build_planted_model(SPEC).hidden(domain_batch(SPEC, seed=11))
    assert torch.allclose(restored, original, atol=1e-6)
    masker.detach()


def test_diagnostics_run():
    _, _, acc_gen = _accs()
    s = scores(acc_gen, "gate_mass")
    gini = specialization_gini(s)
    assert gini.shape == (SPEC.n_layers,) and np.all(gini >= 0)
    report = standing_committee_report(s)
    assert 0 < report[0.5]["mean"] <= SPEC.n_experts


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"PASS {fn.__name__}")
