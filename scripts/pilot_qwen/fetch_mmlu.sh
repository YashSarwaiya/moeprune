#!/bin/bash
# Run from the repo root (on a machine with internet access):
#   bash scripts/pilot_qwen/fetch_mmlu.sh
# Downloads the official MMLU test CSVs and keeps six non-STEM subjects as
# the third-domain probe (knowledge the code/math union was never given).
set -euo pipefail

SUBJECTS="philosophy world_religions jurisprudence high_school_european_history marketing prehistory"

mkdir -p data/mmlu
curl -sL -o /tmp/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
for s in $SUBJECTS; do
  tar -xf /tmp/mmlu_data.tar -C /tmp "data/test/${s}_test.csv"
  mv "/tmp/data/test/${s}_test.csv" data/mmlu/
done
rm -rf /tmp/mmlu_data.tar /tmp/data

python - <<'EOF'
import csv, glob
rows = 0
for f in glob.glob("data/mmlu/*_test.csv"):
    with open(f, newline='', encoding='utf-8') as fh:
        for row in csv.reader(fh):
            assert len(row) == 6 and row[5].strip() in "ABCD", (f, row[:1])
            rows += 1
print(f"MMLU probe OK: {rows} questions across {len(glob.glob('data/mmlu/*.csv'))} subjects")
EOF
