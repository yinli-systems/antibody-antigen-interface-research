#!/usr/bin/env bash
# Development-only pair-to-node aggregation attribution matrix.
#SBATCH --job-name=fc-joint-attr-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --array=0-14%15
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-attr-v1-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-attr-v1-%A_%a.err

set -euo pipefail
data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
repository_root="${FRAMECONTACT_REPOSITORY_ROOT:?FRAMECONTACT_REPOSITORY_ROOT must name the immutable attribution source}"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${data_root}/asep/development-cache"
output_root="${data_root}/asep/development-runs/joint-attribution-dev-v1"
protocol="${repository_root}/protocols/framecontact_asep_joint_attribution_dev_v1.json"
protocol_receipt="${repository_root}/protocols/FRAMECONTACT_ASEP_JOINT_ATTRIBUTION_DEV_V1_RECEIPT.json"
train_panel="${data_root}/asep/panels/train-reciprocal-31.json"
validation_panel="${data_root}/asep/panels/validation-reciprocal-31.json"
cache_sha="69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7"
train_panel_sha="a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1"
validation_panel_sha="10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec"
seeds=(17 42 73 101 137)
experiments=(framecontact_joint_a1_noisy_or framecontact_joint_attribution_mean_probability framecontact_joint_attribution_length_normalized_noisy_or)
task_index="${SLURM_ARRAY_TASK_ID:-${FC_TASK_INDEX:-}}"
if [[ -z "${task_index}" || "${task_index}" -lt 0 || "${task_index}" -ge 15 ]]; then echo "SLURM_ARRAY_TASK_ID must be in 0..14" >&2; exit 2; fi
experiment="${experiments[$((task_index / 5))]}"
seed="${seeds[$((task_index % 5))]}"
run="${output_root}/${experiment}/seed-${seed}"
if [[ -f "${run}/RUN_RECEIPT.json" && -f "${run}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then exit 0; fi
if [[ -e "${run}" ]]; then echo "Partial output exists without complete receipt: ${run}" >&2; exit 2; fi
for required in "${repository_root}" "${cache_root}" "${protocol}" "${protocol_receipt}" "${train_panel}" "${validation_panel}"; do [[ -e "${required}" ]] || { echo "Missing development authority: ${required}" >&2; exit 2; }; done
cd "${repository_root}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${repository_root}/src"
"${python_bin}" scripts/train_framecontact.py \
  --dataset-backend asep --asep-cache-root "${cache_root}" --asep-cache-manifest-sha256 "${cache_sha}" --asep-feature-mode official_esm2_35m \
  --protocol "${protocol}" --protocol-receipt "${protocol_receipt}" --protocol-receipt-sha256 "$(sha256sum "${protocol_receipt}" | awk '{print $1}')" \
  --training-partner-panel "${train_panel}" --training-partner-panel-sha256 "${train_panel_sha}" --validation-partner-panel "${validation_panel}" --validation-partner-panel-sha256 "${validation_panel_sha}" \
  --output-root "${run}" --seed "${seed}" --experiment "${experiment}" --epochs 120 --early-stopping-patience 20 --batch-size 1 --evaluation-batch-size 1 --workers 0 \
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 --dropout 0.1 --attention-kernel reference \
  --learning-rate 0.0003 --weight-decay 0.0001 --contact-loss-weight 1.0 --paratope-loss-weight 0.25 --epitope-loss-weight 0.25 --specificity-loss-weight 0.25 \
  --distogram-loss-weight 0.0 --contact-distance-consistency-loss-weight 0.0 --pair-node-consistency-loss-weight 0.0 --node-spatial-smoothness-loss-weight 0.0 \
  --triangle-loss-weight 0.0 --far-patch-exclusion-loss-weight 0.0 --edm-loss-weight 0.0 --residue-partner-contrastive-loss-weight 0.0 \
  --specificity-margin 0.05 --score-fraction 0.01 --calibration-max-pairs 2000000 --contact-gate-warmup-fraction 0.05 --contact-gate-ramp-fraction 0.10 \
  --checkpoint-wrong-partners --offload-saved-tensors --device cuda --precision bf16
