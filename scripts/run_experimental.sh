#!/usr/bin/env bash
# End-to-end experimental-structure replication: build set -> annotate -> harvest
# -> aggregate -> metrics. Mirrors the AlphaFold atlas pipeline on RCSB structures.
set -e
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export PATH="$PATH:$HOME/anaconda3/envs/dssp/bin"

echo "### 1/5 build experimental set"
python3 scripts/build_experimental_set.py --target 180 --scan-limit 650

echo "### 2/5 annotate (SS8 + SASA from experimental coords)"
python3 scripts/annotate_structures.py --dir data/experimental --workers 8

echo "### 3/5 harvest activations"
python3 scripts/harvest_scaled.py --config config/harvest_experimental.json

echo "### 4/5 aggregate by layer"
python3 scripts/aggregate_activations.py --base activations/experimental

echo "### 5/5 metrics"
python3 scripts/compute_metrics.py --base activations/experimental --tag experimental

echo "### DONE experimental replication"
