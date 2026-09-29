#!/bin/bash
#SBATCH --job-name=ab-corresp-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=/ssd/scxi253/correspondence_upgrade_20260930/attempt1/logs/smoke-%j.out
#SBATCH --error=/ssd/scxi253/correspondence_upgrade_20260930/attempt1/logs/smoke-%j.err
set -euo pipefail
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ROOT=/ssd/scxi253/correspondence_upgrade_20260930
PY=/ssd/scxi253/repr_extract/venv/bin/python3
cd "$ROOT/source"
"$PY" extensions/correspondence_v2/test_core.py > "$ROOT/attempt1/UNIT_TESTS.txt" 2>&1
"$PY" -u extensions/correspondence_v2/run.py freeze --plan "$ROOT/attempt1/PLAN.json"
SHA=$(sha256sum "$ROOT/attempt1/PLAN.json" | cut -d" " -f1)
printf "%s\n" "$SHA" > "$ROOT/attempt1/PLAN.sha256"
"$PY" -u extensions/correspondence_v2/run.py run --plan "$ROOT/attempt1/PLAN.json" --plan-sha256 "$SHA" --task 0 --output "$ROOT/attempt1/out" --smoke
