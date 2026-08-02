#!/bin/bash
#SBATCH --job-name=abf-fsdp2-recovery
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-fsdp2-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-fsdp2-recovery-%j.err

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
COMMON=(--shard "$SHARD_ROOT" --manifest-sha256 "$MANIFEST_SHA256" --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" --micro-batch 4 --target-tokens 16384 --batch-quantum 128 --backend fsdp2)
launch() {
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" "${COMMON[@]}" "$@"
}
launch --output "$RUN_ROOT/continuous" --steps 4
launch --output "$RUN_ROOT/stage1" --steps 2
launch --output "$RUN_ROOT/resumed" --steps 2 --resume-root "$RUN_ROOT/stage1"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/verify_antibody_recovery.py" --continuous "$RUN_ROOT/continuous" --resumed "$RUN_ROOT/resumed" --output "$RUN_ROOT/recovery_receipt.json"
