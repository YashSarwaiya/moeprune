#!/bin/bash
# Run from the repo root (on a machine with internet access):
#   bash scripts/pilot_qwen/predownload.sh
# Downloads the model (~60 GB) + eval sets into $MOEPRUNE_WORKDIR so compute
# jobs can run fully offline (HF_HUB_OFFLINE=1) on clusters whose compute
# nodes lack internet access.
set -euo pipefail

WORKDIR="${MOEPRUNE_WORKDIR:?set MOEPRUNE_WORKDIR to a large scratch dir}"
export HF_HOME=$WORKDIR/hf_cache          # NEVER the default — $HOME is 40 GB
export HF_HUB_ENABLE_HF_TRANSFER=1

# Dedicated venv. If your site provides a torch module, load it first and
# keep --system-site-packages so this venv inherits it.
[ -d "$WORKDIR/k3env" ] || python -m venv --system-site-packages "$WORKDIR/k3env"
source "$WORKDIR/k3env/bin/activate"
# Pin below the v5 MoE refactor: 4.5x keeps Qwen3-MoE's classic structure
# (gate = plain Linear, experts = ModuleList) that our hooks and the masker
# are verified against.
pip install -q -U "transformers>=4.51,<4.56" accelerate huggingface_hub hf_transfer
python -c "import torch, transformers as t; print('torch', torch.__version__, '| transformers', t.__version__)"

echo "downloading Qwen/Qwen3-30B-A3B-Instruct-2507 (~60 GB) to $HF_HOME ..."
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-30B-A3B-Instruct-2507")
print("model cached OK")
EOF

mkdir -p data
curl -sL -o data/HumanEval.jsonl.gz \
  https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
curl -sL -o data/gsm8k_test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
python - <<'EOF'
import gzip, json
n = sum(1 for _ in gzip.open("data/HumanEval.jsonl.gz", "rt"))
m = sum(1 for _ in open("data/gsm8k_test.jsonl"))
assert n == 164 and m > 1000, (n, m)
print(f"eval sets OK: HumanEval {n}, GSM8K {m}")
EOF

echo "predownload complete — ready to sbatch scripts/pilot_qwen/pilot.sbatch"
