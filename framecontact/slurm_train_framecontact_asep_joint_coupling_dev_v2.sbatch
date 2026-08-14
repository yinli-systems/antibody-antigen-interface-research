#!/usr/bin/env bash
#SBATCH --job-name=fc-joint-v2
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --array=0-29%30
#SBATCH --output=fc-joint-v2-%A_%a.out
#SBATCH --error=fc-joint-v2-%A_%a.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_REPOSITORY_ROOT:-${data_root}/asep/joint-coupling-source-v2}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${data_root}/asep/development-cache"
output_root="${data_root}/asep/development-runs/joint-coupling-dev-v2"
protocol="${repository_root}/protocols/framecontact_asep_joint_coupling_dev_v2.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_DEV_V2_RECEIPT.json"
protocol_receipt_sha="d4f63d7b75b3eaa6d90be70cd55663831868dbca1a3f87b38d9b39be11ac04cb"
train_panel="${data_root}/asep/panels/train-reciprocal-31.json"
validation_panel="${data_root}/asep/panels/validation-reciprocal-31.json"
cache_sha="69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7"
train_panel_sha="a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1"
validation_panel_sha="10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec"

seeds=(17 42 73 101 137)
experiments=(
  framecontact_joint_a0_pair_only
  framecontact_joint_a1_noisy_or
  framecontact_joint_a2_independent_heads
  framecontact_joint_a3_coupled_heads
  framecontact_joint_a4_coupled_soft_crop
  framecontact_joint_a5_graph_spatial
)
task_index="${SLURM_ARRAY_TASK_ID:-${FC_TASK_INDEX:-}}"
if [[ -z "${task_index}" ]]; then
  echo "SLURM_ARRAY_TASK_ID or FC_TASK_INDEX is required" >&2
  exit 2
fi
if (( task_index < 0 || task_index >= 30 )); then
  echo "Task index is outside the preregistered 0-29 development matrix" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"

paratope_weight=0.25
epitope_weight=0.25
pair_node_weight=0.0
distogram_weight=0.0
contact_distance_weight=0.0
node_spatial_weight=0.0
case "${experiment}" in
  framecontact_joint_a0_pair_only)
    paratope_weight=0.0
    epitope_weight=0.0
    ;;
  framecontact_joint_a3_coupled_heads|framecontact_joint_a4_coupled_soft_crop)
    pair_node_weight=0.25
    ;;
  framecontact_joint_a5_graph_spatial)
    pair_node_weight=0.25
    node_spatial_weight=0.10
    ;;
esac

run="${output_root}/${experiment}/seed-${seed}"
if [[ -f "${run}/RUN_RECEIPT.json" && -f "${run}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
  exit 0
fi
if [[ -e "${run}" ]]; then
  echo "Partial output exists without a complete receipt: ${run}" >&2
  exit 2
fi
for required in "${repository_root}" "${cache_root}" "${protocol}" "${protocol_receipt}" "${train_panel}" "${validation_panel}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required development authority is missing: ${required}" >&2
    exit 2
  fi
done

cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend asep \
  --asep-cache-root "${cache_root}" \
  --asep-cache-manifest-sha256 "${cache_sha}" \
  --asep-feature-mode official_esm2_35m \
  --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "${protocol_receipt_sha}" \
  --training-partner-panel "${train_panel}" \
  --training-partner-panel-sha256 "${train_panel_sha}" \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 "${validation_panel_sha}" \
  --output-root "${run}" \
  --seed "${seed}" \
  --experiment "${experiment}" \
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
  --paratope-loss-weight "${paratope_weight}" \
  --epitope-loss-weight "${epitope_weight}" \
  --specificity-loss-weight 0.25 \
  --distogram-loss-weight "${distogram_weight}" \
  --contact-distance-consistency-loss-weight "${contact_distance_weight}" \
  --pair-node-consistency-loss-weight "${pair_node_weight}" \
  --node-spatial-smoothness-loss-weight "${node_spatial_weight}" \
  --node-spatial-neighbor-radius 12.0 \
  --node-spatial-neighbor-k 8 \
  --triangle-loss-weight 0.0 \
  --far-patch-exclusion-loss-weight 0.0 \
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
  --checkpoint-wrong-partners \
  --offload-saved-tensors \
  --device cuda \
  --precision bf16
