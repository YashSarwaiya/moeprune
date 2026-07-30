"""Inspect a local safetensors checkpoint directory (run on the storage box
after downloading moonshotai/Kimi-K3).

Reads only the JSON header of each shard (first few KB), so it runs in seconds
on a 1.56 TB checkpoint. Prints:
  - total bytes, and the exact fraction held by routed-expert tensors
    (replaces the UNVERIFIED routed_expert_bytes_fraction in config.py)
  - the observed tensor-name patterns, to fill in prune.KeyMap
  - layer count and expert count sanity check vs config.ModelSpec

Usage: python3 scripts/inspect_checkpoint.py /path/to/Kimi-K3
"""
import collections
import json
import re
import struct
import sys
from pathlib import Path

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1,
               "F8_E5M2": 1, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F4": 0.5, "MXFP4": 0.5}

# Candidate patterns for expert tensors across common MoE naming schemes;
# the real K3 pattern is whichever one matches.
EXPERT_RES = [re.compile(p) for p in (
    r"layers\.(\d+)\..*experts\.(\d+)\.",
    r"blocks\.(\d+)\..*expert_(\d+)\.",
)]


def shard_tensors(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    for name, info in header.items():
        if name == "__metadata__":
            continue
        n = 1
        for d in info["shape"]:
            n *= d
        yield name, n * DTYPE_BYTES.get(info["dtype"], 2)


def main(ckpt_dir: str) -> None:
    shards = sorted(Path(ckpt_dir).glob("*.safetensors"))
    if not shards:
        sys.exit(f"no .safetensors shards under {ckpt_dir}")
    total = expert_bytes = 0
    layers, experts = set(), set()
    prefix_bytes = collections.Counter()
    for shard in shards:
        for name, nbytes in shard_tensors(shard):
            total += nbytes
            prefix_bytes[".".join(name.split(".")[:3])] += nbytes
            for rx in EXPERT_RES:
                m = rx.search(name)
                if m:
                    expert_bytes += nbytes
                    layers.add(int(m.group(1)))
                    experts.add(int(m.group(2)))
                    break
    print(f"shards: {len(shards)}   total: {total/1e12:.3f} TB")
    if layers:
        print(f"routed-expert tensors: {expert_bytes/1e12:.3f} TB "
              f"({expert_bytes/total:.1%})  <- set ModelSpec.routed_expert_bytes_fraction")
        print(f"layers with experts: {len(layers)} (max id {max(layers)})  "
              f"experts per layer: {len(experts)} (max id {max(experts)})")
    else:
        print("NO expert tensors matched known patterns — inspect names below "
              "and extend EXPERT_RES + prune.KeyMap:")
    print("\nlargest tensor-name prefixes:")
    for prefix, b in prefix_bytes.most_common(12):
        print(f"  {b/1e9:>8.1f} GB  {prefix}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
