"""Produce routing logs (.npz) from a calibration set.

Two modes:

  --parse-llamacpp DUMP.jsonl
      Fold a routing dump produced by the patched llama.cpp fork (cheap CPU-box
      path; gate_mass scoring only). See README for the patch.

  --model PATH_OR_HF_ID
      Load with transformers, attach RouterLogger, teacher-forced prefill over
      prompt+reference of each calibration item. Needs hardware that can hold
      the model; intended for the GPU/big-RAM box, or small stand-in models
      when rehearsing the pipeline.

Usage:
  python3 scripts/log_routing.py --calib calibration/coding.jsonl \
      --out logs/coding.npz (--model ... | --parse-llamacpp dump.jsonl)
"""
import argparse
import json
import sys

sys.path.insert(0, "src")

from moeprune.config import ModelSpec
from moeprune.instrument import RouterLogger, parse_llamacpp_log


def iter_texts(calib_path: str):
    with open(calib_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["id"], item["prompt"] + "\n\n" + item.get("reference", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--parse-llamacpp")
    ap.add_argument("--gate-pattern", default=r"layers\.(\d+)\.moe\.gate$")
    ap.add_argument("--expert-pattern", default=r"layers\.(\d+)\.moe\.experts\.(\d+)$")
    args = ap.parse_args()
    spec = ModelSpec()

    if args.parse_llamacpp:
        acc = parse_llamacpp_log(args.parse_llamacpp, spec.n_layers,
                                 spec.n_routed_experts)
        acc.save(args.out)
        print(f"folded llama.cpp dump -> {args.out} "
              f"({acc.tokens_seen} tokens, norms={acc.has_norms})")
        return

    if not args.model:
        sys.exit("need --model or --parse-llamacpp")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    model.eval()
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", spec.n_layers)
    n_experts = getattr(cfg, "n_routed_experts", spec.n_routed_experts)
    top_k = getattr(cfg, "num_experts_per_tok", spec.top_k)
    logger = RouterLogger(model, n_layers, n_experts, top_k,
                          gate_pattern=args.gate_pattern,
                          expert_pattern=args.expert_pattern)
    for item_id, text in iter_texts(args.calib):
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids)                       # prefill only — no generation
        logger.flush(ids.numel())
        print(f"  {item_id}: {ids.numel()} tokens")
    logger.detach()
    logger.acc.save(args.out)
    print(f"saved {args.out} ({logger.acc.tokens_seen} tokens, "
          f"norms={logger.acc.has_norms})")


if __name__ == "__main__":
    main()
