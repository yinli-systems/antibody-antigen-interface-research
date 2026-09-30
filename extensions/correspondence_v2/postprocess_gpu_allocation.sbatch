#!/bin/bash
#SBATCH --job-name=ab-corresp-analyze
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
# This partition rejects CPU-only jobs; analysis itself is CPU work.
#SBATCH --time=00:05:00
#SBATCH --output=/ssd/scxi253/correspondence_upgrade_20260930/attempt1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/correspondence_upgrade_20260930/attempt1/logs/analyze-%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
A=/ssd/scxi253/correspondence_upgrade_20260930/attempt1
S=/ssd/scxi253/correspondence_upgrade_20260930/source
PY=/ssd/scxi253/repr_extract/venv/bin/python3
"$PY" "$A/postprocess/check_identity.py" --plan "$A/PLAN.json" \
 --references "$A/postprocess/HISTORICAL_IDENTITY_REFERENCES.json" \
 --results "$A/out" --output "$A/IDENTITY_PARITY.json"
"$PY" "$S/extensions/correspondence_v2/aggregate.py" --plan "$A/PLAN.json" \
 --results "$A/out" --output "$A/ANALYSIS.json"
