"""Pilot: full domain-pruning pipeline on Qwen3-30B-A3B-Instruct-2507.

Single-GPU, single-job. Loads the model once, then:
  1. logs routing over the coding + general calibration sets (prefill only)
  2. scores experts, detects super experts, builds manifests per sweep ratio
  3. evaluates baseline + each masked-pruned variant on HumanEval (in-domain)
     and GSM8K (out-of-domain)
  4. writes results JSON incrementally + prints the curve

Qwen3-30B-A3B: 128 routed experts/layer, top-8 — the closest public
architecture to K3's fine-grained design. Prior art (MoNE) puts its
GENERAL-pruning limit at ~25%; the hypothesis says domain-restricted pruning
goes far higher.

Usage (inside the SBATCH job):
  python scripts/pilot_qwen/pilot.py --out results/pilot_${SLURM_JOB_ID}
"""
import argparse
import gzip
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from moeprune.instrument import RouterLogger                      # noqa: E402
from moeprune.masking import ExpertMasker                         # noqa: E402
from moeprune.protect import standing_committee_report, super_experts  # noqa: E402
from moeprune.prune import Manifest, build_manifest               # noqa: E402
from moeprune.scoring import scores, specialization_gini          # noqa: E402

GATE_PAT = r"layers\.(\d+)\.mlp\.gate$"
EXPERT_PAT = r"layers\.(\d+)\.mlp\.experts\.(\d+)$"
COMMON_IMPORTS = ("import math\nimport re\nimport itertools\n"
                  "import collections\nfrom typing import *\n")


def log_calibration(model, tok, calib_path, n_layers, n_experts, top_k, device):
    logger = RouterLogger(model, n_layers, n_experts, top_k,
                          gate_pattern=GATE_PAT, expert_pattern=EXPERT_PAT)
    with open(calib_path) as f:
        for line in f:
            item = json.loads(line)
            text = item["prompt"] + "\n\n" + item.get("reference", "")
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=2048).input_ids.to(device)
            with torch.inference_mode():
                model(ids)
            logger.flush(ids.numel())
    logger.detach()
    return logger.acc


def chat_generate(model, tok, contents, max_new, batch_size, device):
    """Greedy batched generation through the chat template."""
    prompts = [tok.apply_chat_template([{"role": "user", "content": c}],
                                       tokenize=False, add_generation_prompt=True)
               for c in contents]
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True)
        print(f"    generated {min(i + batch_size, len(prompts))}/{len(prompts)}",
              flush=True)
    return outs


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def run_humaneval_program(program):
    # Standard HumanEval practice: execute the candidate in a subprocess with
    # a timeout, in a temp dir. Compute node, no privileges — accepted risk
    # for this benchmark's test harnesses.
    try:
        r = subprocess.run([sys.executable, "-c", program], timeout=10,
                           capture_output=True, cwd=tempfile.gettempdir())
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def eval_humaneval(model, tok, problems, batch_size, device):
    asks = [("Complete this Python function. Reply with ONLY the complete "
             "function (signature included) in a single ```python code "
             "block.\n\n```python\n" + p["prompt"] + "\n```") for p in problems]
    gens = chat_generate(model, tok, asks, max_new=512,
                         batch_size=batch_size, device=device)
    programs = []
    for p, g in zip(problems, gens):
        code = extract_code(g)
        if f"def {p['entry_point']}" not in code:
            code = p["prompt"] + "\n" + code   # model returned a body only
        prompt_imports = "\n".join(l for l in p["prompt"].splitlines()
                                   if l.startswith(("import ", "from ")))
        programs.append(COMMON_IMPORTS + prompt_imports + "\n" + code + "\n"
                        + p["test"] + f"\ncheck({p['entry_point']})\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        passed = list(pool.map(run_humaneval_program, programs))
    return sum(passed) / len(passed)


def eval_gsm8k(model, tok, problems, batch_size, device):
    asks = [(p["question"] + "\n\nThink step by step briefly, then give the "
             "final line exactly as: Answer: <number>") for p in problems]
    gens = chat_generate(model, tok, asks, max_new=640,
                         batch_size=batch_size, device=device)
    correct = 0
    for p, g in zip(problems, gens):
        gold = float(p["answer"].split("####")[-1].strip().replace(",", ""))
        nums = re.findall(r"-?[\d,]*\.?\d+", g.replace("$", ""))
        if nums and abs(float(nums[-1].replace(",", "")) - gold) < 1e-4:
            correct += 1
    return correct / len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--out", default="results/pilot")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ratios", default="0.25,0.5,0.65,0.8,0.875")
    ap.add_argument("--random-ratios", default="",
                    help="also eval RANDOM expert selections at these ratios "
                         "(the control: does scored selection beat luck?)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--he-limit", type=int, default=164)
    ap.add_argument("--gsm-limit", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    except TypeError:   # transformers v5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    assert model.dtype == torch.bfloat16, f"loaded as {model.dtype}, not bf16"
    device = model.device
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_experts = cfg.num_experts
    top_k = cfg.num_experts_per_tok
    print(f"loaded in {time.time()-t0:.0f}s: {n_layers} layers, "
          f"{n_experts} experts, top-{top_k}, dtype {model.dtype}", flush=True)

    # Fail fast if this transformers version restructured the MoE block:
    # the masker requires gates to be plain Linear logits over experts.
    gate_re = re.compile(GATE_PAT)
    gates = [m for n, m in model.named_modules() if gate_re.search(n)]
    if len(gates) != n_layers or not all(
            isinstance(g, torch.nn.Linear) and g.out_features == n_experts
            for g in gates):
        sys.exit(f"ABORT: expected {n_layers} Linear gates with "
                 f"{n_experts} outputs, found {len(gates)} matching "
                 f"'{GATE_PAT}'. This transformers version restructured "
                 f"Qwen3-MoE — pin transformers>=4.51,<4.56 and resubmit.")

    # ---- 1. calibration ----------------------------------------------------
    print("logging routing: coding set", flush=True)
    acc_dom = log_calibration(model, tok, REPO / "calibration/coding.jsonl",
                              n_layers, n_experts, top_k, device)
    print("logging routing: general set", flush=True)
    acc_gen = log_calibration(model, tok, REPO / "calibration/general.jsonl",
                              n_layers, n_experts, top_k, device)
    acc_dom.save(out / "coding.npz")
    acc_gen.save(out / "general.npz")

    # ---- 2. scores, protection, manifests ----------------------------------
    s_dom, s_gen = scores(acc_dom, "contrib"), scores(acc_gen, "contrib")
    gini = specialization_gini(s_gen)
    committee = standing_committee_report(s_gen)
    protected = super_experts(s_gen, tau=8.0, top_frac=0.02)
    diag = {
        "gini_mean": float(gini.mean()), "gini_min": float(gini.min()),
        "gini_max": float(gini.max()),
        "committee": {str(k): v for k, v in committee.items()},
        "protected_per_layer": float(np.mean([len(p) for p in protected])),
    }
    print(f"diagnostics: {json.dumps(diag, indent=2)}", flush=True)

    ratios = [float(r) for r in args.ratios.split(",") if r]
    manifests = {}
    for r in ratios:
        m = build_manifest(s_dom, protected, ratio=r, min_keep=2 * top_k,
                           top_k=top_k)
        m.save(out / f"manifest_{int(r*1000)}.json")
        manifests[r] = m

    # Control: same ratios, experts chosen by seeded coin-flip. If scored
    # pruning doesn't beat this, our selection method adds nothing.
    random_ratios = [float(r) for r in args.random_ratios.split(",") if r]
    random_manifests = {}
    rng = np.random.default_rng(0)
    for r in random_ratios:
        n_keep = max(2 * top_k, int(round(n_experts * (1.0 - r))))
        keep = [sorted(rng.choice(n_experts, size=n_keep,
                                  replace=False).tolist())
                for _ in range(n_layers)]
        m = Manifest(ratio=r, method="random", keep=keep,
                     protected=[[] for _ in range(n_layers)])
        m.save(out / f"manifest_random_{int(r*1000)}.json")
        random_manifests[r] = m

    # ---- 3. eval sweep ------------------------------------------------------
    with gzip.open(Path(args.data_dir) / "HumanEval.jsonl.gz", "rt") as f:
        he = [json.loads(l) for l in f][:args.he_limit]
    with open(Path(args.data_dir) / "gsm8k_test.jsonl") as f:
        gsm = [json.loads(l) for l in f][:args.gsm_limit]

    masker = ExpertMasker(model, n_experts, gate_pattern=GATE_PAT)
    results = {"model": args.model, "diagnostics": diag, "points": []}

    def eval_point(label, ratio, kept):
        t = time.time()
        he_score = eval_humaneval(model, tok, he, args.batch, device)
        gsm_score = eval_gsm8k(model, tok, gsm, args.batch, device)
        point = {"label": label, "ratio": ratio, "kept_per_layer": kept,
                 "humaneval_pass1": he_score, "gsm8k_em": gsm_score,
                 "minutes": (time.time() - t) / 60}
        results["points"].append(point)
        (out / "results.json").write_text(json.dumps(results, indent=2))
        print(f"  {label}: HumanEval {he_score:.1%}  GSM8K {gsm_score:.1%} "
              f"({point['minutes']:.0f} min)", flush=True)

    if not args.skip_baseline:
        print("evaluating baseline (no pruning)", flush=True)
        masker.clear()
        eval_point("baseline", 0.0, n_experts)
    for r in ratios:
        m = manifests[r]
        print(f"evaluating ratio {r:.1%} "
              f"(keep {len(m.keep[0])}/{n_experts} per layer)", flush=True)
        masker.apply(m)
        eval_point(f"pruned_{r:.0%}", r, len(m.keep[0]))
    for r in random_ratios:
        m = random_manifests[r]
        print(f"evaluating RANDOM ratio {r:.1%} "
              f"(keep {len(m.keep[0])}/{n_experts} per layer)", flush=True)
        masker.apply(m)
        eval_point(f"random_{r:.0%}", r, len(m.keep[0]))

    # ---- 4. report -----------------------------------------------------------
    print("\n| point | ratio | kept/layer | HumanEval pass@1 | GSM8K EM |")
    print("|---|---|---|---|---|")
    for p in results["points"]:
        print(f"| {p['label']} | {p['ratio']:.1%} | {p['kept_per_layer']} "
              f"| {p['humaneval_pass1']:.1%} | {p['gsm8k_em']:.1%} |")
    print(f"\nresults written to {out}/results.json")


if __name__ == "__main__":
    main()
