#!/usr/bin/env bash
#SBATCH --job-name=fc-distogram
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-distogram-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-distogram-%A_%a.err

set -euo pipefail

seeds=(17 42 73 101 137)
experiments=(framecontact_distogram framecontact_distogram_triangle)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= 10 )); then
  echo "SLURM_ARRAY_TASK_ID is outside the frozen 0-9 matrix" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"
if [[ "$experiment" == framecontact_distogram_triangle ]]; then
  triangle_weight=0.1
else
  triangle_weight=0.0
fi

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/residue-head-source-v4"
source_manifest="$base/source-manifests/residue-head-source-v4.json"
run_root="$base/runs/distogram-exploratory-v1"
output_root="$run_root/$experiment/seed-$seed"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3

if [[ -f "$output_root/RUN_RECEIPT.json" && -f "$output_root/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
  exit 0
fi
if [[ -e "$output_root" ]]; then
  echo "Partial output exists without a complete receipt: $output_root" >&2
  exit 2
fi

cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$source_root/src"

"$python_bin" scripts/freeze_training_source_tree.py verify \
  --source-root "$source_root" \
  --manifest "$source_manifest" \
  --manifest-sha256 7a8f335baca281e0daf4020708221252cb02adf09b4a090e761907e075178195

"$python_bin" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "$base/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "$base/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --pair-geometry-cache-root "$base/sabdab2/pair-geometry-development-v1" \
  --pair-geometry-manifest-sha256 5924697eaffb6ae023834cd72bd179aefd95a6d8890bfca5ce7e2dd9feba41b2 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "$base/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "$source_root/protocols/framecontact_distogram_exploratory_v1.json" \
  --protocol-receipt "$source_root/protocols/FRAMECONTACT_DISTOGRAM_EXPLORATORY_V1_RECEIPT.json" \
  --protocol-receipt-sha256 7ca2933dca175af2f49463d05fe4be8a90ae791edddf6da9062a9bb9fc7b944e \
  --source-manifest "$source_manifest" \
  --source-manifest-sha256 7a8f335baca281e0daf4020708221252cb02adf09b4a090e761907e075178195 \
  --training-partner-panel "$base/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "$base/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "$output_root" \
  --seed "$seed" \
  --experiment "$experiment" \
  --epochs 200 \
  --early-stopping-patience 20 \
  --batch-size 1 \
  --evaluation-batch-size 1 \
  --workers 4 \
  --d-model 128 \
  --num-heads 8 \
  --num-self-layers 2 \
  --num-rbf 16 \
  --neighbor-k 32 \
  --neighbor-radius 16.0 \
  --sequence-window 32 \
  --dropout 0.1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --contact-loss-weight 1.0 \
  --paratope-loss-weight 0.25 \
  --epitope-loss-weight 0.25 \
  --specificity-loss-weight 0.25 \
  --distogram-loss-weight 0.5 \
  --triangle-loss-weight "$triangle_weight" \
  --specificity-margin 0.05 \
  --score-fraction 0.01 \
  --calibration-max-pairs 2000000 \
  --triangle-neighbor-radius 12.0 \
  --triangle-neighbor-k 8 \
  --triangle-slack 0.5 \
  --antigen-patch-top-seeds 1 \
  --antigen-patch-radius 12.0 \
  --offload-saved-tensors \
  --device cuda \
  --precision bf16
