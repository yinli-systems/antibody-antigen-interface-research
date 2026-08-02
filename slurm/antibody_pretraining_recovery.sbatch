#!/bin/bash
#SBATCH --job-name=abf-recovery
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-recovery-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4
COMMON=(--shard "$SHARD_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" --micro-batch 8 --target-tokens 4096 --batch-quantum 32)
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" "${COMMON[@]}" --output "$RUN_ROOT/continuous" --steps 4
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" "${COMMON[@]}" --output "$RUN_ROOT/stage1" --steps 2
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" "${COMMON[@]}" --output "$RUN_ROOT/resumed" --steps 2 --resume-root "$RUN_ROOT/stage1"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/verify_antibody_recovery.py" --continuous "$RUN_ROOT/continuous" --resumed "$RUN_ROOT/resumed" --output "$RUN_ROOT/recovery_receipt.json"
