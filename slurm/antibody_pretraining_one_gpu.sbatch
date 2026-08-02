#!/bin/bash
#SBATCH --job-name=abf-one-gpu
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/abf-one-gpu-%j.out
#SBATCH --error=/tmp/abf-one-gpu-%j.err

set -euo pipefail
umask 077
PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
SHARD_ROOT=${ABF_SHARD_ROOT:?ABF_SHARD_ROOT is required}
MANIFEST_SHA256=${ABF_MANIFEST_SHA256:?ABF_MANIFEST_SHA256 is required}
RUN_ROOT=$OUTPUT_ROOT/slurm-$SLURM_JOB_ID
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite the run root" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$SOURCE_ROOT/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 2 > "$OUTPUT_ROOT/gpu-$SLURM_JOB_ID.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/run_antibody_pretraining.py" \
  --shard "$SHARD_ROOT" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --model-family "$SOURCE_ROOT/protocols/antibody_model_family_v1.json" \
  --training-protocol "$SOURCE_ROOT/protocols/antibody_training_v1.json" \
  --output "$RUN_ROOT" \
  --steps 3 \
  --micro-batch 8 \
  --target-tokens 4096 \
  --batch-quantum 32
