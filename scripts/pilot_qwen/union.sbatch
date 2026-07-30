#!/bin/bash
#SBATCH --job-name=moeprune-union
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=04:00:00
#SBATCH --output=moeprune-union_%j.log

# Union experiment: code-pruned vs math-pruned vs union-of-both at 65%,
# each evaluated on HumanEval AND GSM8K. Tests whether domain-pruned
# variants compose — the open question no published work answers.

# Submit with your site's scheduler options, e.g.:
#   sbatch --account=ACCT --qos=QOS --partition=PART thisfile.sbatch
set -euo pipefail
# Set these for your site, or export before submitting.
WORKDIR="${MOEPRUNE_WORKDIR:?set MOEPRUNE_WORKDIR to a large scratch dir}"
REPO="${MOEPRUNE_REPO:-$PWD}"

export HF_HOME="$WORKDIR/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source "$WORKDIR/k3env/bin/activate"
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; \
print(torch.cuda.get_device_name(0))"

python scripts/pilot_qwen/union.py --out "results/union_${SLURM_JOB_ID}"
