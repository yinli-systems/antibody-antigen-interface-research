#!/usr/bin/env bash
#SBATCH --job-name=abf30b-sable-chain-map
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-chain-map-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-chain-map-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly authority=${campaign}/authority/sable_label_free_v1
readonly protocol=${authority}/ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1.json
readonly receipt=${authority}/ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1_RECEIPT.json
readonly input_plan=${sable}/structure_prediction/af3_label_free_v1/AF3_LABEL_FREE_INPUT_PLAN.json
readonly structures=${sable}/structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_MANIFEST.json
readonly python_bin=${root}/envs/oas-imgt-anarcii-2.0.8-torch-2.7.1-cpu-py3.12/bin/python

test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha256=$("${python_bin}" - "${receipt}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['frozen_before_chain_typing'] is True; print(x['protocol_sha256'])
PY
)
printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  c82c12f08c87b42d21aee31cc555373ffbc4211415d9aff302444baefc971ad6 "${input_plan}" \
  795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259 "${structures}" \
  | sha256sum --check --strict
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" "${root}/tools/abf30b_framecontact_v1/build_abf30b_sable_chain_map_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
  --input-plan "${input_plan}" --structure-manifest "${structures}" \
  --output "${authority}/ABF30B_SABLE_CHAIN_MAP_V1.json" \
  --receipt-output "${authority}/ABF30B_SABLE_CHAIN_MAP_V1_RECEIPT.json"
