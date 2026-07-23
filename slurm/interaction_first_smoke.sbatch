#!/bin/bash
#SBATCH --job-name=interaction-smoke
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_interaction_first.py
OUTPUT=$DEPLOY_ROOT/results/sabdab2/interaction_first/smoke.json

mkdir -p "$(dirname "$OUTPUT")" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" "$SCRIPT" smoke --seed 42 --output "$OUTPUT"
