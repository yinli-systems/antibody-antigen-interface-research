#!/bin/bash
#SBATCH --job-name=saprot-full
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
[[ $(sha256sum "$ROOT/models/saprot650/pytorch_model.bin" | cut -d" " -f1) == 513c39088804b74309d961174d5fc7aeabfb9b79d2814cdc7e465d4fe4b9f216 ]]
[[ $(sha256sum "$ROOT/models/saprot650/vocab.txt" | cut -d" " -f1) == d4289b3ff55d50b1aa33a3141ec344fc2a69cb3dca43aeefe9fccfee371ffd28 ]]

export PYTHONPATH="$ROOT/pylibs:$PWD/src:$PWD" PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK=${SLURM_ARRAY_TASK_ID:?}
if (( TASK < 16 )); then SIDE=antibody; SHARD=$TASK; else SIDE=antigen; SHARD=$((TASK-16)); fi
"$PY" -u extensions/frontier_multimodel_v1/structural_plm.py extract \
 --model-kind saprot --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" \
 --tokens "$ROOT/tokens/saprot_v2" --model-path "$ROOT/models/saprot650" \
 --model-revision d9b9ad00ef61c0990e611b2b43f2231c7de24b38 --layers 0 8 16 24 33 \
 --side "$SIDE" --shard-index "$SHARD" --shard-count 16 --output "$ATTEMPT_ROOT/output/features"
