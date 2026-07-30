"""Checkpoint surgery: rewrite a safetensors checkpoint per a pruning manifest.

Pure CPU/disk work — runs shard-by-shard on the storage box, peak memory one
shard (~16 GB). Writes a new shard set, updated weight-map index, and a
config.json with the reduced expert count noted.

The router row-slice in prune.apply_to_state_dict assumes the gate weight's
leading dim indexes experts — verify once against the shapes printed by
inspect_checkpoint.py before the paid run.

Usage:
  python3 scripts/prune_checkpoint.py --ckpt /data/Kimi-K3 \
      --manifest manifests/keep_80pct_pruned.json --out /data/Kimi-K3-code-80
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

from moeprune.prune import KeyMap, Manifest, apply_to_state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert-pattern",
                    default=r"^(.*layers\.(\d+)\.moe\.experts\.)(\d+)(\..+)$")
    ap.add_argument("--gate-pattern",
                    default=r"^.*layers\.(\d+)\.moe\.gate\.(weight|bias)$")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    manifest = Manifest.load(args.manifest)
    keymap = KeyMap(expert_pattern=args.expert_pattern,
                    gate_patterns=(args.gate_pattern,))
    src, dst = Path(args.ckpt), Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    weight_map = {}
    for shard in sorted(src.glob("*.safetensors")):
        sd = load_file(str(shard))
        new_sd = apply_to_state_dict(sd, manifest, keymap)
        if new_sd:
            save_file(new_sd, str(dst / shard.name))
            weight_map.update({k: shard.name for k in new_sd})
        dropped = len(sd) - len(new_sd)
        print(f"{shard.name}: {len(sd)} tensors -> {len(new_sd)} "
              f"({dropped} pruned)")

    index = {"metadata": {"pruning_manifest": args.manifest,
                          "ratio": manifest.ratio},
             "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(index))
    for aux in src.glob("*.json"):
        if "index" not in aux.name:
            shutil.copy(aux, dst / aux.name)
    n_keep = len(manifest.keep[0])
    print(f"done. NOTE: edit {dst}/config.json n_routed_experts -> {n_keep} "
          f"(per-layer counts in the manifest if they differ)")


if __name__ == "__main__":
    main()
