#!/bin/bash
#SBATCH --job-name=epiformer-verify
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00:20:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
DATA_ROOT=$DEPLOY_ROOT/data/epiformer/asep/m3epi
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
RUN_DIR=$DEPLOY_ROOT/reports/epiformer-verify-$SLURM_JOB_ID

mkdir -p "$RUN_DIR"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH

/usr/bin/time -v "$PYTHON_BIN" "$DEPLOY_ROOT/jobs/epiformer_verify.py" \
  --dataset "$DATA_ROOT/res_graph_tensor_esm2_650m.pkl" \
  --shard-dir "$DATA_ROOT/res_graph_tensor_esm2_650m_shards" \
  --report "$RUN_DIR/report.json"

sha256sum "$RUN_DIR/report.json" > "$RUN_DIR/sha256.txt"
