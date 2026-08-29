#!/usr/bin/env bash
#SBATCH --job-name=oas-paired-650m-train-canary-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/training-canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/training-canary-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE_ROOT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v8/source
readonly ABF_SOURCE_ROOT=/ssd/scxi253/model_deployment/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1
readonly CAMPAIGN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1
readonly SHARD_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821/oas_paired_development_clean_composite_v1
readonly RUN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/650m-training-canary-cdr3-ce-v1
readonly RESULT=${RUN_ROOT}/OAS_PAIRED_650M_TRAINING_CANARY_CDR3_CE_RESULT_V1.json
readonly TRAINER_SHA256=ba214d618d83e6cbd1aa88a9128c7807ac0b7184f63e01705eee0ef7e5c9b837
readonly PAIRED_SHA256=fcfb69f016681a8d801db213e49f6ff61252f6882d28eb56753cac3dbbdac58a
readonly PROTOCOL_SHA256=e37d8c31736f980e464bcaa6156292380ca0c0520b36760f1db3485aa1f74689

test -x "${PYTHON}"
test ! -e "${RUN_ROOT}"
test -f "${SHARD_ROOT}/MANIFEST.json"

readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden training-canary allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi

cd "${SOURCE_ROOT}"
[[ "$(sha256sum train_650m_paired_canary.py | awk '{print $1}')" == "${TRAINER_SHA256}" ]]
[[ "$(sha256sum paired_pretraining.py | awk '{print $1}')" == "${PAIRED_SHA256}" ]]
[[ "$(sha256sum PROTOCOL_650M_PAIRED_TRAINING_CANARY_CDR3_CE_V1.json | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

mkdir -p "${RUN_ROOT}"
"${PYTHON}" train_650m_paired_canary.py \
  --abf-source-root "${ABF_SOURCE_ROOT}" \
  --checkpoint-root "${CAMPAIGN_ROOT}/segment-000035/run/checkpoint" \
  --checkpoint-receipt "${CAMPAIGN_ROOT}/segment-000035/run/receipt.json" \
  --model-family "${ABF_SOURCE_ROOT}/protocols/antibody_model_family_v1.json" \
  --shard-root "${SHARD_ROOT}" \
  --protocol "${SOURCE_ROOT}/PROTOCOL_650M_PAIRED_TRAINING_CANARY_CDR3_CE_V1.json" \
  --output "${RESULT}"

sha256sum "${RESULT}"
