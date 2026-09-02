#!/usr/bin/env bash
#SBATCH --job-name=sable-open-labels
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/open-bound-labels-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/open-bound-labels-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly data=${root}/data/external/sable_v0_2_0_label_free
readonly tools_root=${root}/tools/sable_external_v1
readonly script=${tools_root}/freeze_sable_v0_2_0_bound_label_opening_v1.py
readonly script_sha256=79a42afd326cfe4751ac58534dc3d503f989b01a05dd375929b1688b10032674
readonly measurement_source_manifest=${tools_root}/SABLE_V0_2_0_MEASUREMENT_SOURCE_MANIFEST_V1.json
readonly measurement_source_manifest_sha256=d67dd8580e4a0e9d4b039223cf2f583e6b48c71f21f33b49b08985e059f9f918
readonly measurement_source_receipt=${tools_root}/SABLE_V0_2_0_MEASUREMENT_SOURCE_MANIFEST_V1_RECEIPT.json
readonly measurement_source_receipt_sha256=5eb14f91690f2465bf1a3569230fe7523ce22e596a92c18fb2a7027c3a42f882
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
echo "${script_sha256}  ${script}" | sha256sum --check --strict -
echo "${measurement_source_manifest_sha256}  ${measurement_source_manifest}" | sha256sum --check --strict -
echo "${measurement_source_receipt_sha256}  ${measurement_source_receipt}" | sha256sum --check --strict -
mkdir -p "${sable}/authority"

"${python_bin}" "${script}" \
  --external-plan "${tools_root}/SABLE_V0_2_0_EXTERNAL_EXECUTION_PLAN_V1.json" \
  --screen-result "${sable}/label_free_gate/SABLE_V0_2_0_LABEL_FREE_SCREEN_RESULT.json" \
  --screen-receipt "${sable}/label_free_gate/SABLE_V0_2_0_LABEL_FREE_SCREEN_RECEIPT.json" \
  --structure-cache-root "${sable}/structure_cache/predicted_label_free_v1" \
  --embedding-root "${sable}/embeddings/esm2_t33_650m_label_free_v1" \
  --checkpoint-binding "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1.json" \
  --checkpoint-binding-receipt "${sable}/authority/SABLE_V0_2_0_NO_DISTANCE_CHECKPOINT_BINDING_V1_RECEIPT.json" \
  --measurement-source-manifest "${measurement_source_manifest}" \
  --measurement-source-receipt "${measurement_source_receipt}" \
  --output "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1.json" \
  --receipt-output "${sable}/authority/SABLE_V0_2_0_BOUND_LABEL_OPENING_AUTHORIZATION_V1_RECEIPT.json"
