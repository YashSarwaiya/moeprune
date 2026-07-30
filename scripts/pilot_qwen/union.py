"""Union experiment: can two domain-pruned variants be recombined?

Builds three keep-sets at the same pruning ratio:
  CODE  — experts scored on the coding calibration set
  MATH  — experts scored on the math calibration set
  UNION — per-layer union of the two

and evaluates each on BOTH benchmarks (HumanEval = coding metric, GSM8K =
math metric). The question: does the union hold both skills at a size far
below two separate models — because the "standing committee" is shared?

Nothing in the literature measures this composition (as of 2026-07); this is
the most original claim the pilot model can support.

Usage: python scripts/pilot_qwen/union.py --out results/union_$SLURM_JOB_ID
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot import (GATE_PAT, eval_gsm8k, eval_humaneval,       # noqa: E402
                   log_calibration)
from moeprune.masking import ExpertMasker                       # noqa: E402
from moeprune.protect import super_experts                      # noqa: E402
from moeprune.prune import Manifest, build_manifest             # noqa: E402
from moeprune.scoring import scores                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--out", default="results/union")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ratio", type=float, default=0.65,
                    help="per-domain pruning ratio (65% = the pilot's money point)")
    ap.add_argument("--he-limit", type=int, default=164)
    ap.add_argument("--gsm-limit", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    device = model.device
    cfg = model.config
    n_layers, n_experts = cfg.num_hidden_layers, cfg.num_experts
    top_k = cfg.num_experts_per_tok

    accs = {}
    for name in ("coding", "math", "general"):
        print(f"logging routing: {name} set", flush=True)
        accs[name] = log_calibration(model, tok, REPO / f"calibration/{name}.jsonl",
                                     n_layers, n_experts, top_k, device)
        accs[name].save(out / f"{name}.npz")

    protected = super_experts(scores(accs["general"], "contrib"), top_frac=0.02)
    m_code = build_manifest(scores(accs["coding"], "contrib"), protected,
                            ratio=args.ratio, min_keep=2 * top_k, top_k=top_k)
    m_math = build_manifest(scores(accs["math"], "contrib"), protected,
                            ratio=args.ratio, min_keep=2 * top_k, top_k=top_k)
    union_keep = [sorted(set(c) | set(m))
                  for c, m in zip(m_code.keep, m_math.keep)]
    m_union = Manifest(ratio=args.ratio, method="union",
                       keep=union_keep, protected=m_code.protected)
    for name, m in (("code", m_code), ("math", m_math), ("union", m_union)):
        m.save(out / f"manifest_{name}.json")

    # Overlap accounting — the scientific payload even before any evals.
    sizes = {
        "code": float(np.mean([len(k) for k in m_code.keep])),
        "math": float(np.mean([len(k) for k in m_math.keep])),
        "union": float(np.mean([len(k) for k in union_keep])),
        "overlap": float(np.mean([len(set(c) & set(m))
                                  for c, m in zip(m_code.keep, m_math.keep)])),
    }
    sizes["union_vs_sum"] = sizes["union"] / (sizes["code"] + sizes["math"])
    print(f"keep sizes/layer: {json.dumps(sizes, indent=2)}", flush=True)

    with gzip.open(Path(args.data_dir) / "HumanEval.jsonl.gz", "rt") as f:
        he = [json.loads(l) for l in f][:args.he_limit]
    with open(Path(args.data_dir) / "gsm8k_test.jsonl") as f:
        gsm = [json.loads(l) for l in f][:args.gsm_limit]

    masker = ExpertMasker(model, n_experts, gate_pattern=GATE_PAT)
    results = {"model": args.model, "ratio": args.ratio, "sizes": sizes,
               "points": []}
    for label, manifest in (("code_pruned", m_code), ("math_pruned", m_math),
                            ("union", m_union)):
        print(f"evaluating {label} "
              f"(keep ~{np.mean([len(k) for k in manifest.keep]):.0f}"
              f"/{n_experts} per layer)", flush=True)
        masker.apply(manifest)
        t = time.time()
        he_score = eval_humaneval(model, tok, he, args.batch, device)
        gsm_score = eval_gsm8k(model, tok, gsm, args.batch, device)
        results["points"].append({
            "label": label,
            "kept_per_layer": float(np.mean([len(k) for k in manifest.keep])),
            "humaneval_pass1": he_score, "gsm8k_em": gsm_score,
            "minutes": (time.time() - t) / 60})
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"  {label}: HumanEval {he_score:.1%}  GSM8K {gsm_score:.1%}",
              flush=True)

    print("\n| variant | kept/layer | HumanEval | GSM8K |")
    print("|---|---|---|---|")
    for p in results["points"]:
        print(f"| {p['label']} | {p['kept_per_layer']:.0f} "
              f"| {p['humaneval_pass1']:.1%} | {p['gsm8k_em']:.1%} |")
    print(f"\nunion size vs sum of parts: {sizes['union_vs_sum']:.0%} "
          f"(overlap {sizes['overlap']:.0f} experts/layer)")
    print(f"results written to {out}/results.json")


if __name__ == "__main__":
    main()
