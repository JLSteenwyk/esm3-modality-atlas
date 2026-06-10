#!/usr/bin/env bash
# Orchestrate the diverse multi-organism scale run end-to-end:
#   (fetch already launched) -> annotate -> harvest -> aggregate -> metrics
set -e
cd "$(dirname "$0")/.."
export PATH="$PATH:$HOME/anaconda3/envs/dssp/bin"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=data/diverse/pipeline.log
echo "=== diverse pipeline start $(date +%H:%M) ===" | tee -a "$LOG"

# 1. wait for the structure fetch to finish (summary reports the full request)
until python3 -c "import json,sys; d=json.load(open('data/diverse/logs/fetch_summary.json')); sys.exit(0 if d.get('requested',0)>5000 else 1)" 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M)] fetch done: $(ls data/diverse/structures/*.pdb | wc -l) structures" | tee -a "$LOG"

# 2. annotate SS8+SASA (parallel)
python3 scripts/annotate_structures.py --dir data/diverse --workers 16 2>&1 | grep -vE "invalid start|bad reference" | tail -3 | tee -a "$LOG"

# 3. harvest 6 conditions x 48 layers
python3 scripts/harvest_scaled.py --config config/harvest_diverse.json 2>&1 | grep -vE "FutureWarning|with torch|Fetching|it/s\]$" | tail -4 | tee -a "$LOG"

# 4. aggregate + metrics
python3 scripts/aggregate_activations.py --base activations/diverse 2>&1 | tail -2 | tee -a "$LOG"
python3 scripts/compute_metrics.py --base activations/diverse --tag diverse 2>&1 | tail -2 | tee -a "$LOG"

echo "=== DIVERSE PIPELINE DONE $(date +%H:%M) ===" | tee -a "$LOG"
