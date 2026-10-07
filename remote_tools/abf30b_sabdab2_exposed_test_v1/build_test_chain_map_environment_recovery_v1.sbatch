#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-chain-r1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/chain-map-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/chain-map-recovery-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly protocol=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json
readonly receipt=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json
readonly input_root=${campaign}/label_free_inputs_v1
readonly structure_root=${campaign}/data/test_structure_cache_v1
readonly sequence_view=${input_root}/SABDAB2_TEST_SEQUENCE_VIEW_V1.jsonl
readonly sequence_receipt=${input_root}/TEST_INPUT_PLAN_V1_CHAIN_MAP_ADAPTER.json
readonly chain_map=${campaign}/authority/SABDAB2_TEST_ABF30B_CHAIN_MAP_V1.json
readonly chain_receipt=${campaign}/authority/SABDAB2_TEST_ABF30B_CHAIN_MAP_V1_RECEIPT.json
readonly python_bin=${root}/envs/oas-imgt-anarcii-2.0.8-torch-2.7.1-cpu-py3.12/bin/python
test "${SLURM_JOB_PARTITION}" = gpu_4090
read -r protocol_sha sequence_sha sequence_receipt_sha structure_sha < <("${python_bin}" - "${protocol}" "${receipt}" "${sequence_view}" "${sequence_receipt}" "${structure_root}/STRUCTURE_CACHE_MANIFEST.json" <<'PY'
import hashlib,json,sys
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
p,r,s,sr,st=sys.argv[1:]; ps=sha(p); assert json.load(open(r))['protocol_sha256']==ps; a=json.load(open(sr)); assert a['task_labels_read'] is False; print(ps,sha(s),sha(sr),sha(st))
PY
)
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" "${tools}/build_test_chain_map_v1.py" \
  --pinned-builder "${root}/tools/abf30b_framecontact_v1/build_abf30b_chain_map_v1.py" \
  --sequence-view-sha256 "${sequence_sha}" --sequence-receipt-sha256 "${sequence_receipt_sha}" \
  --structure-manifest-sha256 "${structure_sha}" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha}" \
  --sequence-view "${sequence_view}" --sequence-receipt "${sequence_receipt}" \
  --structure-root "${structure_root}" --output "${chain_map}" --receipt-output "${chain_receipt}" --batch-size 64
