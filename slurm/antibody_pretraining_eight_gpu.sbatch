#!/bin/bash
#SBATCH --job-name=abf-eight-gpu
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-eight-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-eight-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
BACKEND=${ABF_BACKEND:?ABF_BACKEND is required}
if [[ "$BACKEND" != "ddp" && "$BACKEND" != "fsdp2" ]]; then exit 2; fi
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID-$BACKEND
test ! -e "$RUN_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$SOURCE_ROOT/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
  "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" \
  --shard "$SHARD_ROOT" --manifest-sha256 "$MANIFEST_SHA256" \
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" \
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" \
  --output "$RUN_ROOT" --steps 3 --micro-batch 4 --target-tokens 16384 \
  --batch-quantum 128 --backend "$BACKEND" --skip-checkpoint
