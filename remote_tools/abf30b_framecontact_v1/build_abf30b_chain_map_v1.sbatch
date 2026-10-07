#!/usr/bin/env bash
#SBATCH --job-name=abf30b-chain-map
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/chain-map-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/chain-map-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly tools=${root}/tools/abf30b_framecontact_v1
readonly authority=${campaign}/authority
readonly protocol=${authority}/ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1.json
readonly protocol_receipt=${authority}/ABF30B_FRAMECONTACT_TRANSFER_PROTOCOL_V1_RECEIPT.json
readonly structure_root=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1/data/structure_cache_v1
readonly sequence_root=${root}/data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1
readonly python_bin=${root}/envs/oas-imgt-anarcii-2.0.8-torch-2.7.1-cpu-py3.12/bin/python

test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha256=$("${python_bin}" - "${protocol_receipt}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['frozen_before_chain_typing'] is True
print(x['protocol_sha256'])
PY
)
echo "${protocol_sha256}  ${protocol}" | sha256sum --check --strict -
mkdir -p "${campaign}/logs"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tools}/build_abf30b_chain_map_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
  --sequence-view "${sequence_root}/sabdab2_train_validation_sequences_v1.jsonl" \
  --sequence-receipt "${sequence_root}/SABDAB2_TRAIN_VALIDATION_SEQUENCE_VIEW_V1_RECEIPT.json" \
  --structure-root "${structure_root}" \
  --output "${authority}/SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1.json" \
  --receipt-output "${authority}/SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1_RECEIPT.json" \
  --batch-size 64
