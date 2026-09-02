#!/usr/bin/env bash
#SBATCH --job-name=abf30b-dcp-canary
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/dcp-canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/dcp-canary-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly tools=${root}/tools/abf30b_framecontact_v1
readonly authority=${campaign}/authority
readonly pretrain_source=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/src
readonly protocol=${authority}/ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json
readonly protocol_receipt=${authority}/ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json
readonly chain_map=${authority}/SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1.json
readonly chain_receipt=${authority}/SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1_RECEIPT.json
readonly family=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/protocols/antibody_model_family_v1.json
readonly checkpoint=${root}/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1/segment-000069/run/checkpoint
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
read -r protocol_sha256 chain_map_sha256 < <("${python_bin}" - "${protocol_receipt}" "${chain_receipt}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
assert p['frozen_before_dcp_restore'] is True
assert c['status']=='PASS_LABEL_FREE_ANARCII_CHAIN_TYPING'
print(p['protocol_sha256'],c['chain_map_sha256'])
PY
)
echo "${protocol_sha256}  ${protocol}" | sha256sum --check --strict -
echo "${chain_map_sha256}  ${chain_map}" | sha256sum --check --strict -
printf '%s  %s\n' \
  'f33e90400f6e7a55a434adbc72534fc7c971b7e60b946763c410287f1cc412c3' "${pretrain_source}/antibody_pretraining/model.py" \
  '270a90b6c092a2890d0d84209578d4ba1e9b8c871c5377b57b5c72e3af24e274' "${pretrain_source}/antibody_pretraining/tokenizer.py" \
  | sha256sum --check --strict
export PYTHONPATH=${pretrain_source}
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tools}/canary_abf30b_dcp_embedding_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
  --chain-map "${chain_map}" --chain-map-sha256 "${chain_map_sha256}" \
  --model-family "${family}" --checkpoint "${checkpoint}" \
  --output "${authority}/ABF30B_DCP_EMBEDDING_CANARY_V1.json" \
  --receipt-output "${authority}/ABF30B_DCP_EMBEDDING_CANARY_V1_RECEIPT.json"
