#!/bin/bash
#SBATCH --job-name=moeprune-skeptic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=05:00:00
#SBATCH --output=moeprune-skeptic_%j.log

# The three missing controls for the union result:
#   1. third-domain probe (MMLU non-STEM) for the code∪math union
#   2. random keep-sets size-matched to the union
#   3. hard domain pair: code ∪ humanities

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

python scripts/pilot_qwen/skeptic.py --out "results/skeptic_${SLURM_JOB_ID}"
