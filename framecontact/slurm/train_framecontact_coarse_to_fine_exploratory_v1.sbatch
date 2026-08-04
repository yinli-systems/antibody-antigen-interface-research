#!/usr/bin/env bash
#SBATCH --job-name=fc-ctf-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --array=0-19%5
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-ctf-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-ctf-v1-%A_%a.err

# Post-hoc development-only Plan-B matrix. No frozen/external test material is
# accepted by this entry point. The exact source and protocol must be frozen
# and authenticated before submission.

set -euo pipefail

seeds=(17 42 73 101 137)
experiments=(
  framecontact_icg_m4_interaction_gate
  framecontact_plan_b_soft_crop
  framecontact_plan_b_soft_crop_joint
  framecontact_plan_b_soft_crop_far_patch
)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= 20 )); then
  echo "SLURM_ARRAY_TASK_ID is outside the frozen 0-19 matrix" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"

distogram_weight=0.0
consistency_weight=0.0
far_patch_weight=0.0
pair_geometry_args=()
case "$experiment" in
  framecontact_plan_b_soft_crop_joint)
    distogram_weight=0.25
    consistency_weight=0.1
    ;;
  framecontact_plan_b_soft_crop_far_patch)
    distogram_weight=0.25
    consistency_weight=0.1
    far_patch_weight=0.05
    ;;
esac
if [[ "$distogram_weight" != 0.0 ]]; then
  pair_geometry_args=(
    --pair-geometry-cache-root
    /ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2/pair-geometry-development-v1
    --pair-geometry-manifest-sha256
    5924697eaffb6ae023834cd72bd179aefd95a6d8890bfca5ce7e2dd9feba41b2
  )
fi

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/coarse-to-fine-source-v1"
source_manifest="$base/source-manifests/coarse-to-fine-source-v1.json"
run_root="$base/runs/coarse-to-fine-exploratory-v1"
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
  --manifest-sha256 "${COARSE_TO_FINE_V1_SOURCE_MANIFEST_SHA256:?set from the source freeze receipt}"

"$python_bin" scripts/train_framecontact.py \
  --dataset-backend sabdab2 \
  --structure-cache-root "$base/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "$base/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  "${pair_geometry_args[@]}" \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "$base/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --protocol "$source_root/protocols/framecontact_coarse_to_fine_exploratory_v1.json" \
  --protocol-receipt "$source_root/protocols/FRAMECONTACT_COARSE_TO_FINE_EXPLORATORY_V1_RECEIPT.json" \
  --protocol-receipt-sha256 "${COARSE_TO_FINE_V1_PROTOCOL_RECEIPT_SHA256:?set from the protocol receipt}" \
  --source-manifest "$source_manifest" \
  --source-manifest-sha256 "$COARSE_TO_FINE_V1_SOURCE_MANIFEST_SHA256" \
  --training-partner-panel "$base/partner_metadata_antigen_fold_v3/train-reciprocal-31.json" \
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \
  --validation-partner-panel "$base/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --output-root "$output_root" \
  --seed "$seed" \
  --experiment "$experiment" \
  --epochs 120 \
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
  --attention-kernel reference \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --contact-loss-weight 1.0 \
  --paratope-loss-weight 0.25 \
  --epitope-loss-weight 0.25 \
  --specificity-loss-weight 0.25 \
  --distogram-loss-weight "$distogram_weight" \
  --contact-distance-consistency-loss-weight "$consistency_weight" \
  --triangle-loss-weight 0.0 \
  --far-patch-exclusion-loss-weight "$far_patch_weight" \
  --edm-loss-weight 0.0 \
  --residue-partner-contrastive-loss-weight 0.0 \
  --specificity-margin 0.05 \
  --score-fraction 0.01 \
  --calibration-max-pairs 2000000 \
  --contact-gate-warmup-fraction 0.05 \
  --contact-gate-ramp-fraction 0.10 \
  --coarse-contact-crop-floor 0.05 \
  --coarse-contact-crop-temperature 1.0 \
  --coarse-contact-crop-strength 1.0 \
  --antigen-patch-top-seeds 1 \
  --antigen-patch-radius 12.0 \
  --far-patch-minimum-separation 20.0 \
  --far-patch-residue-k 16 \
  --far-patch-slack 0.5 \
  --checkpoint-wrong-partners \
  --offload-saved-tensors \
  --device cuda \
  --precision bf16
