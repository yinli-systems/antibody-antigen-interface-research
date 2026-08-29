#!/usr/bin/env bash
#SBATCH --job-name=fc-topology-expand
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/asep/results/training-logs/fc-topology-expansion-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/asep/results/training-logs/fc-topology-expansion-%A_%a.err

set -euo pipefail

data_root="/ssd/scxi253/model_deployment/data/framecontact_20260730"
source_root="${data_root}/topology_routing_source_v1"
python_bin="/ssd/scxi253/model_deployment/envs/contact/bin/python3.10"
cache_root="${data_root}/asep/development-cache"
run_root="${data_root}/asep/development-runs/topology-routing-v1/topology_routed_multiscale"
training_panel="${data_root}/asep/panels/train-reciprocal-31.json"
validation_panel="${data_root}/asep/panels/validation-reciprocal-31.json"
protocol="${source_root}/protocols/framecontact_topology_routing_exploratory_v1.json"
protocol_receipt="${source_root}/protocols/FRAMECONTACT_TOPOLOGY_ROUTING_EXPLORATORY_V1_RECEIPT.json"
source_manifest="${source_root}/SOURCE_SHA256SUMS.txt"

expected_source_manifest_sha256="9b94bd32c1536ac79a800a7b731ce7b249aa702217eba5b813d276142c017054"
expected_cache_manifest_sha256="69cc62f0d77858b38508821bf9d56504dd397787a37dad62fbd1a5072a348cc7"
expected_protocol_receipt_sha256="b3a7ae8488e6249ccc527267d7afe41d527860e49e756d56ca12c7975ca5ff31"
expected_training_panel_sha256="a0c5f3e8ea1a0edf7cd9a2043b326ee3be7b054c81dff9fbfcb510db6f2969d1"
expected_validation_panel_sha256="10f50aa122568d4ed1a8261637f2e807f2702997a1aa5d9925983d37cc0d83ec"

actual_source_manifest_sha256="$(sha256sum "${source_manifest}" | awk '{print $1}')"
if [[ "${actual_source_manifest_sha256}" != "${expected_source_manifest_sha256}" ]]; then
  echo "Immutable source manifest hash mismatch" >&2
  exit 2
fi
(cd "${source_root}" && sha256sum --quiet -c SOURCE_SHA256SUMS.txt)

seeds=(42 73 101 137)
task_index="${SLURM_ARRAY_TASK_ID}"
if (( task_index < 0 || task_index >= ${#seeds[@]} )); then
  echo "SLURM_ARRAY_TASK_ID is outside the registered expansion" >&2
  exit 2
fi
seed="${seeds[$task_index]}"
output_root="${run_root}/seed-${seed}"

if [[ -f "${output_root}/RUN_RECEIPT.json" && -f "${output_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" ]]; then
  exit 0
fi
if [[ -e "${output_root}" ]]; then
  echo "Partial output exists without a complete receipt: ${output_root}" >&2
  exit 2
fi

cd "${source_root}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${source_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${python_bin}" scripts/train_framecontact_topology_routing.py \
  --dataset-backend asep \
  --asep-cache-root "${cache_root}" \
  --asep-cache-manifest-sha256 "${expected_cache_manifest_sha256}" \
  --asep-feature-mode official_esm2_35m \
  --protocol "${protocol}" \
  --protocol-receipt "${protocol_receipt}" \
  --protocol-receipt-sha256 "${expected_protocol_receipt_sha256}" \
  --training-partner-panel "${training_panel}" \
  --training-partner-panel-sha256 "${expected_training_panel_sha256}" \
  --validation-partner-panel "${validation_panel}" \
  --validation-partner-panel-sha256 "${expected_validation_panel_sha256}" \
  --output-root "${output_root}" \
  --seed "${seed}" \
  --experiment topology_routed_multiscale \
  --epochs 100 \
  --early-stopping-patience 15 \
  --batch-size 2 \
  --evaluation-batch-size 8 \
  --workers 4 \
  --d-model 128 \
  --num-heads 8 \
  --num-self-layers 2 \
  --num-rbf 16 \
  --neighbor-k 32 \
  --neighbor-radius 16 \
  --sequence-window 32 \
  --dropout 0.1 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --contact-loss-weight 1 \
  --paratope-loss-weight 0.25 \
  --epitope-loss-weight 0.25 \
  --specificity-loss-weight 0.25 \
  --specificity-margin 0.05 \
  --score-fraction 0.01 \
  --calibration-max-pairs 2000000 \
  --precision bf16 \
  --device cuda
