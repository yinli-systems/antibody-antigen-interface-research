#!/usr/bin/env bash
# Development-only six-condition joint coupling matrix on the epitope-ratio split.
# This intentionally does not open frozen/official/external test roles.
#SBATCH --job-name=fc-joint-ratio-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --array=0-29%30
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-ratio-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-ratio-v1-%A_%a.err

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_REPOSITORY_ROOT:?FRAMECONTACT_REPOSITORY_ROOT must name the immutable ratio source}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${FRAMECONTACT_ASEP_RATIO_CACHE_ROOT:-/ssd/scxi253/model_deployment/data/epiformer/asep/framecontact-ratio-compat-development-cache-20260811}"
output_root="${data_root}/asep/development-runs/joint-coupling-ratio-v1"
protocol="${repository_root}/protocols/framecontact_asep_joint_coupling_ratio_v1.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_COUPLING_RATIO_V1_RECEIPT.json"
ratio_panel_root="${FRAMECONTACT_ASEP_RATIO_PANEL_ROOT:-/ssd/scxi253/model_deployment/data/epiformer/asep/framecontact-ratio-partner-panels-20260811}"
train_panel="${ratio_panel_root}/train-reciprocal-31.json"
validation_panel="${ratio_panel_root}/validation-reciprocal-31.json"
cache_sha="57ca5bcaf3b15ddca3b80a684d9f5cff9944c75e8749563287de2971bdef0c8e"
train_panel_sha="4242fced77cf114fc11330f773cee2753317ef80d4f61aa6533c217c2f982ecb"
validation_panel_sha="52c65f17d3c3b501de6a760bbf779505ddf730f55c874316b71eddc55a33d93a"

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
if [[ -z "${task_index}" || "${task_index}" -lt 0 || "${task_index}" -ge 30 ]]; then
  echo "SLURM_ARRAY_TASK_ID must be in 0..29" >&2
  exit 2
fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"
run="${output_root}/${experiment}/seed-${seed}"
if [[ -f "${run}/RUN_RECEIPT.json" && -f "${run}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then exit 0; fi
if [[ -e "${run}" ]]; then echo "Partial output exists without a complete receipt: ${run}" >&2; exit 2; fi
for required in "${repository_root}" "${cache_root}" "${protocol}" "${protocol_receipt}" "${train_panel}" "${validation_panel}"; do
  [[ -e "${required}" ]] || { echo "Missing development authority: ${required}" >&2; exit 2; }
done

paratope_weight=0.25
epitope_weight=0.25
pair_node_weight=0.0
node_spatial_weight=0.0
case "${experiment}" in
  framecontact_joint_a0_pair_only) paratope_weight=0.0; epitope_weight=0.0 ;;
  framecontact_joint_a3_coupled_heads|framecontact_joint_a4_coupled_soft_crop) pair_node_weight=0.25 ;;
  framecontact_joint_a5_graph_spatial) pair_node_weight=0.25; node_spatial_weight=0.10 ;;
esac

cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend asep --asep-cache-root "${cache_root}" \
  --asep-cache-manifest-sha256 "${cache_sha}" --asep-feature-mode official_esm2_35m \
  --protocol "${protocol}" --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "$(sha256sum "${protocol_receipt}" | awk '{print $1}')" \
  --training-partner-panel "${train_panel}" --training-partner-panel-sha256 "${train_panel_sha}" \
  --validation-partner-panel "${validation_panel}" --validation-partner-panel-sha256 "${validation_panel_sha}" \
  --output-root "${run}" --seed "${seed}" --experiment "${experiment}" \
  --epochs 120 --early-stopping-patience 20 --batch-size 1 --evaluation-batch-size 1 --workers 0 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 --neighbor-k 32 --neighbor-radius 16.0 \
  --sequence-window 32 --dropout 0.1 --attention-kernel reference --learning-rate 0.0003 --weight-decay 0.0001 \
  --contact-loss-weight 1.0 --paratope-loss-weight "${paratope_weight}" --epitope-loss-weight "${epitope_weight}" \
  --specificity-loss-weight 0.25 --distogram-loss-weight 0.0 --contact-distance-consistency-loss-weight 0.0 \
  --pair-node-consistency-loss-weight "${pair_node_weight}" --node-spatial-smoothness-loss-weight "${node_spatial_weight}" \
  --triangle-loss-weight 0.0 --far-patch-exclusion-loss-weight 0.0 --edm-loss-weight 0.0 \
  --residue-partner-contrastive-loss-weight 0.0 --specificity-margin 0.05 --score-fraction 0.01 \
  --calibration-max-pairs 2000000 --contact-gate-warmup-fraction 0.05 --contact-gate-ramp-fraction 0.10 \
  --coarse-contact-crop-floor 0.05 --coarse-contact-crop-temperature 1.0 --coarse-contact-crop-strength 1.0 \
  --checkpoint-wrong-partners --offload-saved-tensors --device cuda --precision bf16
