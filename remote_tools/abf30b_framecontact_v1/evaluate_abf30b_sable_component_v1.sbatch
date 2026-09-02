#!/usr/bin/env bash
#SBATCH --job-name=abf30b-sable-eval
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-eval-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-eval-%A_%a.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly tools_root=${root}/tools/abf30b_framecontact_v1
readonly script=${tools_root}/evaluate_abf30b_sable_component_v1.py
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly seeds=(17 29 43)

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo '2ee62cdc49bee78dc5ee6856cd23e62d0fc6807ea844f9082c920b451bca3aea' "${script}" | sha256sum --check --strict -
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  readonly arm=baseline
  readonly seed=${seeds[SLURM_ARRAY_TASK_ID]}
else
  readonly arm=edge_topk2_supervised
  readonly seed=${seeds[SLURM_ARRAY_TASK_ID-3]}
fi
readonly run_root=${campaign}/sable_measurement_v1/components/${arm}/seed-${seed}
readonly authority=${campaign}/authority/sable_measurement_v1
readonly binding=${authority}/frozen_binding_v1
mkdir -p "${run_root}"
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${script}" \
  --measurement-plan "${authority}/ABF30B_SABLE_MEASUREMENT_PLAN_V1.json" \
  --authorization "${binding}/ABF30B_SABLE_MEASUREMENT_AUTHORIZATION_V1.json" \
  --authorization-receipt "${binding}/ABF30B_SABLE_MEASUREMENT_AUTHORIZATION_V1_RECEIPT.json" \
  --checkpoint-binding "${binding}/ABF30B_SABLE_CHECKPOINT_BINDING_V1.json" \
  --checkpoint-binding-receipt "${binding}/ABF30B_SABLE_CHECKPOINT_BINDING_V1_RECEIPT.json" \
  --structure-root "${sable}/structure_cache/predicted_label_free_v1" \
  --embedding-root "${campaign}/embeddings/sable_abf30b_antibody_esm2_antigen_v1" \
  --label-root "${sable}/bound_labels_once_v1" \
  --arm "${arm}" --seed "${seed}" \
  --output "${run_root}/SABLE_COMPONENT_RESULT.json" \
  --receipt-output "${run_root}/SABLE_COMPONENT_RECEIPT.json"
