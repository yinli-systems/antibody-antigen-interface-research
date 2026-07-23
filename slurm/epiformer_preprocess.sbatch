#!/bin/bash
#SBATCH --job-name=epiformer-prep
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
JOB_ROOT=$DEPLOY_ROOT/jobs
DATA_ROOT=$DEPLOY_ROOT/data/epiformer/asep
SOURCE_ROOT=$DEPLOY_ROOT/src/epiformer
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
RUN_DIR=$DEPLOY_ROOT/reports/epiformer-prep-$SLURM_JOB_ID

mkdir -p "$DEPLOY_ROOT/logs" "$RUN_DIR" "$DEPLOY_ROOT/cache/huggingface" "$DEPLOY_ROOT/cache/torch" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH
export HF_HOME=$DEPLOY_ROOT/cache/huggingface
export TORCH_HOME=$DEPLOY_ROOT/cache/torch
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$RUN_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

echo "job_id=$SLURM_JOB_ID"
echo "node=$(hostname)"
date --iso-8601=seconds
nvidia-smi

/usr/bin/time -v "$PYTHON_BIN" "$JOB_ROOT/epiformer_embed_antiberty.py" --data-root "$DATA_ROOT"
/usr/bin/time -v "$PYTHON_BIN" "$JOB_ROOT/epiformer_embed_650m.py" --data-root "$DATA_ROOT"
/usr/bin/time -v "$PYTHON_BIN" "$JOB_ROOT/epiformer_build_650m.py" --source-root "$SOURCE_ROOT" --data-root "$DATA_ROOT"

date --iso-8601=seconds
sha256sum \
  "$DATA_ROOT/antibody/antiberty_embeddings/asep_antiberty_embeddings.pt" \
  "$DATA_ROOT/antigen/plm_embeddings/ag_esm2_embeddings_asep.pt" \
  "$DATA_ROOT/m3epi/res_graph_tensor_esm2_650m.pkl" > "$RUN_DIR/sha256.txt"
