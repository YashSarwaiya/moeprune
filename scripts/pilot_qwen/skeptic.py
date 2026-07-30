"""Skeptic-proofing run: the three controls the union result still needs.

1. THIRD-DOMAIN PROBE — evaluate the code∪math union on non-STEM MMLU
   subjects it was never calibrated for. A true two-skill model should sit
   well BELOW baseline there; matching baseline would mean we accidentally
   rebuilt a general model.
2. SIZE-MATCHED RANDOM — random keep-sets with exactly the union's per-layer
   sizes. Kills the "67 experts is simply enough" objection.
3. HARD DOMAIN PAIR — code ∪ humanities (distant skills, unlike code+math).
   Composition must hold: high HumanEval, high MMLU-humanities — while GSM8K
   (neither domain) stays low, doubling as a built-in third-domain probe.

Each variant is evaluated on all three benchmarks. Baseline gets MMLU only
(its HumanEval/GSM8K numbers are already measured: 91.5 / 99.3).

Usage: python scripts/pilot_qwen/skeptic.py --out results/skeptic_$SLURM_JOB_ID
"""
import argparse
import csv
import glob
import gzip
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot import (GATE_PAT, chat_generate, eval_gsm8k,        # noqa: E402
                   eval_humaneval, log_calibration)
from moeprune.masking import ExpertMasker                       # noqa: E402
from moeprune.protect import super_experts                      # noqa: E402
from moeprune.prune import Manifest, build_manifest             # noqa: E402
from moeprune.scoring import scores                             # noqa: E402


def load_mmlu(data_dir: str, limit: int) -> list[dict]:
    items = []
    for path in sorted(glob.glob(str(Path(data_dir) / "mmlu" / "*_test.csv"))):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                items.append({"q": row[0], "opts": row[1:5], "gold": row[5].strip()})
    rng = np.random.default_rng(7)
    rng.shuffle(items)
    return items[:limit]


def eval_mmlu(model, tok, items, batch_size, device):
    asks = [(p["q"] + "\n\n" + "\n".join(f"{l}. {o}" for l, o in
             zip("ABCD", p["opts"])) +
             "\n\nAnswer with only the letter (A, B, C, or D).")
            for p in items]
    gens = chat_generate(model, tok, asks, max_new=8,
                         batch_size=batch_size, device=device)
    correct = 0
    for p, g in zip(items, gens):
        m = (re.search(r"[Aa]nswer(?:\s+is)?[:\s]*\(?([ABCD])\)?", g)
             or re.search(r"\b([ABCD])\b", g))
        if m and m.group(1) == p["gold"]:
            correct += 1
    return correct / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--out", default="results/skeptic")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ratio", type=float, default=0.65)
    ap.add_argument("--he-limit", type=int, default=164)
    ap.add_argument("--gsm-limit", type=int, default=150)
    ap.add_argument("--mmlu-limit", type=int, default=300)
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
    for name in ("coding", "math", "humanities", "general"):
        print(f"logging routing: {name} set", flush=True)
        accs[name] = log_calibration(model, tok, REPO / f"calibration/{name}.jsonl",
                                     n_layers, n_experts, top_k, device)
        accs[name].save(out / f"{name}.npz")

    protected = super_experts(scores(accs["general"], "contrib"), top_frac=0.02)

    def dom_manifest(name):
        return build_manifest(scores(accs[name], "contrib"), protected,
                              ratio=args.ratio, min_keep=2 * top_k, top_k=top_k)

    m_code, m_math, m_hum = (dom_manifest(n) for n in
                             ("coding", "math", "humanities"))
    union_cm = Manifest(ratio=args.ratio, method="union_code_math",
                        keep=[sorted(set(c) | set(m))
                              for c, m in zip(m_code.keep, m_math.keep)],
                        protected=m_code.protected)
    union_ch = Manifest(ratio=args.ratio, method="union_code_hum",
                        keep=[sorted(set(c) | set(h))
                              for c, h in zip(m_code.keep, m_hum.keep)],
                        protected=m_code.protected)
    # Random control with EXACTLY the code∪math union's per-layer sizes.
    rng = np.random.default_rng(1)
    rand_matched = Manifest(ratio=args.ratio, method="random_size_matched",
                            keep=[sorted(rng.choice(n_experts, size=len(k),
                                                    replace=False).tolist())
                                  for k in union_cm.keep],
                            protected=[[] for _ in range(n_layers)])
    for name, m in (("code", m_code), ("math", m_math), ("hum", m_hum),
                    ("union_cm", union_cm), ("union_ch", union_ch),
                    ("rand_matched", rand_matched)):
        m.save(out / f"manifest_{name}.json")

    he_data = None
    with gzip.open(Path(args.data_dir) / "HumanEval.jsonl.gz", "rt") as f:
        he_data = [json.loads(l) for l in f][:args.he_limit]
    with open(Path(args.data_dir) / "gsm8k_test.jsonl") as f:
        gsm_data = [json.loads(l) for l in f][:args.gsm_limit]
    mmlu_data = load_mmlu(args.data_dir, args.mmlu_limit)
    print(f"MMLU probe: {len(mmlu_data)} non-STEM questions", flush=True)

    masker = ExpertMasker(model, n_experts, gate_pattern=GATE_PAT)
    results = {"model": args.model, "ratio": args.ratio, "points": []}

    def eval_point(label, manifest, benchmarks=("he", "gsm", "mmlu")):
        if manifest is None:
            masker.clear()
            kept = n_experts
        else:
            masker.apply(manifest)
            kept = float(np.mean([len(k) for k in manifest.keep]))
        print(f"evaluating {label} (keep ~{kept:.0f}/{n_experts})", flush=True)
        t = time.time()
        point = {"label": label, "kept_per_layer": kept}
        if "he" in benchmarks:
            point["humaneval_pass1"] = eval_humaneval(model, tok, he_data,
                                                      args.batch, device)
        if "gsm" in benchmarks:
            point["gsm8k_em"] = eval_gsm8k(model, tok, gsm_data,
                                           args.batch, device)
        if "mmlu" in benchmarks:
            point["mmlu_hum_em"] = eval_mmlu(model, tok, mmlu_data,
                                             args.batch, device)
        point["minutes"] = (time.time() - t) / 60
        results["points"].append(point)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"  {label}: " + "  ".join(
            f"{k}={v:.1%}" for k, v in point.items()
            if k.endswith(("pass1", "_em"))), flush=True)

    eval_point("baseline", None, benchmarks=("mmlu",))          # HE/GSM known
    eval_point("union_code_math", union_cm, benchmarks=("mmlu",))  # HE/GSM known
    eval_point("random_size_matched", rand_matched)
    eval_point("hum_pruned", m_hum)
    eval_point("union_code_hum", union_ch)

    print("\n| point | kept/layer | HumanEval | GSM8K | MMLU-hum |")
    print("|---|---|---|---|---|")
    for p in results["points"]:
        def fmt(key):
            return f"{p[key]:.1%}" if key in p else "—"
        print(f"| {p['label']} | {p['kept_per_layer']:.0f} "
              f"| {fmt('humaneval_pass1')} | {fmt('gsm8k_em')} "
              f"| {fmt('mmlu_hum_em')} |")
    print(f"\nresults written to {out}/results.json")


if __name__ == "__main__":
    main()
