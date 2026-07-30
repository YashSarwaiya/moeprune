#!/bin/bash
#SBATCH --job-name=moeprune-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=06:00:00
#SBATCH --output=moeprune-pilot_%j.log

# Submit with your site's scheduler options, e.g.:
#   sbatch --account=ACCT --qos=QOS --partition=PART thisfile.sbatch
set -euo pipefail
# Set these for your site, or export before submitting.
WORKDIR="${MOEPRUNE_WORKDIR:?set MOEPRUNE_WORKDIR to a large scratch dir}"
REPO="${MOEPRUNE_REPO:-$PWD}"

export HF_HOME="$WORKDIR/hf_cache"
export HF_HUB_OFFLINE=1                 # everything was predownloaded
export TOKENIZERS_PARALLELISM=false

source "$WORKDIR/k3env/bin/activate"
cd "$REPO"

# Blackwell sanity check before committing hours
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; \
print(torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))"

PYTHONPATH=src python tests/test_e2e_synthetic.py   # 20s CPU insurance

python scripts/pilot_qwen/pilot.py --out "results/pilot_${SLURM_JOB_ID}"
