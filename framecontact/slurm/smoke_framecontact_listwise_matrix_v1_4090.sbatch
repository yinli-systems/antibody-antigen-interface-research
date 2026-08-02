#!/usr/bin/env bash
#SBATCH --job-name=fc-listwise-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-listwise-smoke-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-listwise-smoke-%A_%a.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/listwise-source-v1"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3
evidence_root="$base/listwise-evidence-v1"
negative_counts=(1 4 8)
negative_count="${negative_counts[$SLURM_ARRAY_TASK_ID]}"
receipt="$evidence_root/LISTWISE_K${negative_count}_T1_GPU_SMOKE.json"

mkdir -p "$evidence_root" "$base/slurm-logs"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$source_root/src"
cd "$source_root"

loss_mode=listwise_contrastive
direction_aggregation=smooth_max
checkpoint_arguments=()
if [[ "$negative_count" == 1 ]]; then
  loss_mode=pairwise_softplus
  direction_aggregation=mean
else
  checkpoint_arguments+=(--checkpoint-wrong-partners)
fi

"$python_bin" scripts/stress_sabdab2_three_way_training_step.py \
  --structure-root "$base/sabdab2/structure_cohort_v2" \
  --contact-root "$base/sabdab2/contact_cohort_v2" \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --split-manifest "$base/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --training-panel "$base/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --output "$receipt" \
  --experiment framecontact_distogram \
  --specificity-negative-count "$negative_count" \
  --specificity-loss-mode "$loss_mode" \
  --specificity-temperature 1.0 \
  --specificity-direction-aggregation "$direction_aggregation" \
  --specificity-direction-temperature 0.1 \
  "${checkpoint_arguments[@]}" \
  --offload-saved-tensors

sha256sum "$receipt"
