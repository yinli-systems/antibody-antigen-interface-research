#!/bin/bash
#SBATCH --job-name=prosst-quant-full
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
set -euo pipefail
: "${PROGRAM_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}"
ROOT="$PROGRAM_ROOT"
PLAN="$ATTEMPT_ROOT/PLAN.json"
PY=/ssd/scxi253/repr_extract/venv/bin/python3
cd "$ROOT/source"
ACTUAL=$(sha256sum "$PLAN" | cut -d" " -f1)
[[ "$ACTUAL" == "$PLAN_SHA256" ]]
[[ $(git -C "$ROOT/external/ProSST" rev-parse HEAD) == 13bd56495641a84cd5475eaef7aa98b14fdc6ed5 ]]

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/pylibs_prosst_official_v1:$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK=${SLURM_ARRAY_TASK_ID:?}
if (( TASK < 32 )); then SIDE=antibody; SHARD=$TASK; else SIDE=antigen; SHARD=$((TASK-32)); fi
"$PY" -u extensions/frontier_multimodel_v1/structural_plm.py prepare-prosst \
 --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" \
 --prosst-source "$ROOT/external/ProSST" --prosst-source-commit 13bd56495641a84cd5475eaef7aa98b14fdc6ed5 \
 --side "$SIDE" --shard-index "$SHARD" --shard-count 32 --device cuda --output "$ATTEMPT_ROOT/output/prosst_tokens"
