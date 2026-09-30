#!/bin/bash
#SBATCH --job-name=prosst-full
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
set -euo pipefail
: "${PROGRAM_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}"
ROOT="$PROGRAM_ROOT"
PLAN="$ATTEMPT_ROOT/PLAN.json"
PY=/ssd/scxi253/repr_extract/venv/bin/python3
cd "$ROOT/source"
ACTUAL=$(sha256sum "$PLAN" | cut -d" " -f1)
[[ "$ACTUAL" == "$PLAN_SHA256" ]]

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/pylibs_prosst_official_v1:$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK=${SLURM_ARRAY_TASK_ID:?}
if (( TASK < 16 )); then SIDE=antibody; SHARD=$TASK; else SIDE=antigen; SHARD=$((TASK-16)); fi
"$PY" -u extensions/frontier_multimodel_v1/structural_plm.py extract \
 --model-kind prosst --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" \
 --tokens "$ATTEMPT_ROOT/output/prosst_tokens" --model-path "$ROOT/models/prosst2048" \
 --model-revision e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8 --layers 0 3 6 9 12 \
 --side "$SIDE" --shard-index "$SHARD" --shard-count 16 --output "$ATTEMPT_ROOT/output/features"
