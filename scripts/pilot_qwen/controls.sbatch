#!/bin/bash
#SBATCH --job-name=moeprune-controls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --output=moeprune-controls_%j.log

# Follow-up to the main pilot sweep:
#   - random-pruning control at 50/65/80% (does scored selection beat luck?)
#   - 70% and 75% scored points (localize the coding cliff between 65 and 80)
# Baseline is skipped — we already have it (91.5% / 99.3%).

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

python scripts/pilot_qwen/pilot.py \
  --skip-baseline \
  --ratios 0.70,0.75 \
  --random-ratios 0.50,0.65,0.80 \
  --out "results/controls_${SLURM_JOB_ID}"
