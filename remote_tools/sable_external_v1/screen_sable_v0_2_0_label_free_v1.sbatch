#!/usr/bin/env bash
#SBATCH --job-name=sable-label-free-screen
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/label-free-screen-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sable_v0_2_0_framecontact_external_v1/logs/label-free-screen-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tools=${root}/tools/sable_external_v1
readonly data=${root}/data/external/sable_v0_2_0_label_free
readonly development=${root}/data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1
readonly output=${root}/results/development/sable_v0_2_0_framecontact_external_v1/label_free_gate/SABLE_V0_2_0_LABEL_FREE_SCREEN_RESULT.json
readonly receipt=${root}/results/development/sable_v0_2_0_framecontact_external_v1/label_free_gate/SABLE_V0_2_0_LABEL_FREE_SCREEN_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly plan_sha256=6187210253ca07b2d6c5e11e017552485bbb2699bc3772f67d0ee6ad9657d9a4
readonly source_sha256=17ed317cf18671a902105363721359684450a5cb796ba349ec976728260acd8d

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
test ! -e "${receipt}"
echo "${plan_sha256}  ${tools}/SABLE_V0_2_0_LABEL_FREE_SCREEN_PLAN.json" | sha256sum --check --strict -
echo "${source_sha256}  ${tools}/screen_sable_v0_2_0_label_free_v1.py" | sha256sum --check --strict -
mkdir -p "$(dirname "${output}")" "${root}/results/development/sable_v0_2_0_framecontact_external_v1/logs"

"${python_bin}" "${tools}/screen_sable_v0_2_0_label_free_v1.py" \
  --plan "${tools}/SABLE_V0_2_0_LABEL_FREE_SCREEN_PLAN.json" \
  --metadata "${data}/sable_v0.2.0_metadata.csv" \
  --development-sequences "${development}/sabdab2_train_validation_sequences_v1.jsonl" \
  --development-receipt "${development}/SABDAB2_TRAIN_VALIDATION_SEQUENCE_VIEW_V1_RECEIPT.json" \
  --output "${output}" \
  --receipt-output "${receipt}" \
  --workers 4
