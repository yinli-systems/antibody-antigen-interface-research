#!/usr/bin/env bash
#SBATCH --job-name=analyze-bilateral-topk-train
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_training_ablation_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_bilateral_topk_training_ablation_v1/logs/analyze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/sabdab2_bilateral_topk_training_ablation_v1
readonly output=${campaign}/analysis_v1
test ! -e "${output}"
"${root}/envs/contact/bin/python3.10" \
  "${root}/tools/bilateral_topk_training_ablation_v1/analyze_bilateral_topk_training_ablation_v1.py" \
  --campaign-root "${campaign}" --output-root "${output}"
