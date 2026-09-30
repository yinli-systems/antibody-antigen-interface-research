#!/bin/bash
#SBATCH --job-name=structural-audit
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
set -euo pipefail
: "${PROGRAM_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}"
ROOT="$PROGRAM_ROOT"
PLAN="$ATTEMPT_ROOT/PLAN.json"
PY=/ssd/scxi253/repr_extract/venv/bin/python3
cd "$ROOT/source"
ACTUAL=$(sha256sum "$PLAN" | cut -d" " -f1)
[[ "$ACTUAL" == "$PLAN_SHA256" ]]

export PYTHONPATH="$ROOT/pylibs:$PWD/src:$PWD" PYTHONHASHSEED=0
"$PY" -u extensions/frontier_multimodel_v1/structural_plm_full.py audit \
 --plan "$PLAN" --plan-sha256 "$PLAN_SHA256" --output "$ATTEMPT_ROOT/COVERAGE_AUDIT.json"
