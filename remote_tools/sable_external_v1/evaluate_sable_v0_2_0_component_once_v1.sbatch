#!/usr/bin/env bash
#SBATCH --job-name=sable-eval-once
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/evaluate-once-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/evaluate-once-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/evaluate_sable_v0_2_0_component_once_v1.py
readonly script_sha256=4e203b58d9d54c2559aaf27c1d3684e8b49bec9f9c9e322e88bf615ec82f9677
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly seeds=(17 29 43)

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  readonly arm=baseline
  readonly seed=${seeds[SLURM_ARRAY_TASK_ID]}
else
  readonly arm=edge_topk2_supervised
  readonly seed=${seeds[SLURM_ARRAY_TASK_ID-3]}
fi
readonly run_root=${sable}/runs_external_once_v1/${arm}/seed-${seed}
mkdir -p "${run_root}"

"${python_bin}" "${script}" \
  --external-plan "${tools_root}/SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json" \
  --authorization "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1.json" \
  --authorization-receipt "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1_RECEIPT.json" \
  --checkpoint-binding "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1.json" \
  --checkpoint-binding-receipt "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1_RECEIPT.json" \
  --structure-root "${sable}/structure_cache/predicted_label_free_v1" \
  --embedding-root "${sable}/embeddings/esm2_t33_650m_label_free_v1" \
  --label-root "${sable}/bound_labels_once_v1" \
  --arm "${arm}" \
  --seed "${seed}" \
  --output "${run_root}/SABLE_COMPONENT_RESULT.json" \
  --receipt-output "${run_root}/SABLE_COMPONENT_RECEIPT.json"
