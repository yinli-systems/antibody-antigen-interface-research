#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-embed-r1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/embedding-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/embedding-recovery-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly base=${root}/results/development/abf30b_framecontact_v1
readonly protocol=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json
readonly receipt=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json
readonly input_root=${campaign}/label_free_inputs_v1
readonly structure_root=${campaign}/data/test_structure_cache_v1
readonly sequence_view=${input_root}/SABDAB2_TEST_SEQUENCE_VIEW_V1.jsonl
readonly sequence_receipt=${input_root}/TEST_INPUT_PLAN_V1_CHAIN_MAP_ADAPTER.json
readonly chain_map=${campaign}/authority/SABDAB2_TEST_ABF30B_CHAIN_MAP_V1.json
readonly chain_receipt=${campaign}/authority/SABDAB2_TEST_ABF30B_CHAIN_MAP_V1_RECEIPT.json
readonly embedding_root=${campaign}/embeddings/sabdab2_test_abf30b_antibody_esm2_antigen_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
test -n "${CUDA_VISIBLE_DEVICES:-}"
read -r protocol_sha sequence_sha sequence_receipt_sha structure_sha canary_sha < <("${python_bin}" - "${protocol}" "${receipt}" "${sequence_view}" "${sequence_receipt}" "${structure_root}/STRUCTURE_CACHE_MANIFEST.json" "${base}/authority/ABF30B_DCP_EMBEDDING_CANARY_V1.json" <<'PY'
import hashlib,json,sys
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
p,r,s,sr,st,c=sys.argv[1:]; raw=open(p,'rb').read(); d=hashlib.sha256(raw).hexdigest(); assert json.load(open(r))['protocol_sha256']==d
adapter=json.load(open(sr)); assert adapter['task_labels_read'] is False and adapter['task_labels_loaded'] is False
print(d,sha(s),sha(sr),sha(st),sha(c))
PY
)
export PYTHONPATH=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/src
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${python_bin}" "${tools}/build_test_chain_map_v1.py" \
  --pinned-builder "${root}/tools/abf30b_framecontact_v1/build_abf30b_chain_map_v1.py" \
  --sequence-view-sha256 "${sequence_sha}" --sequence-receipt-sha256 "${sequence_receipt_sha}" \
  --structure-manifest-sha256 "${structure_sha}" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha}" \
  --sequence-view "${sequence_view}" --sequence-receipt "${sequence_receipt}" \
  --structure-root "${structure_root}" --output "${chain_map}" --receipt-output "${chain_receipt}" --batch-size 64
readonly chain_sha=$(sha256sum "${chain_map}" | awk '{print $1}')
"${python_bin}" "${tools}/materialize_abf30b_test_embeddings_v1.py" \
  --pinned-materializer "${root}/tools/abf30b_framecontact_v1/materialize_abf30b_sabdab2_embeddings_v1.py" \
  --sequence-view-sha256 "${sequence_sha}" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha}" \
  --chain-map "${chain_map}" --chain-map-sha256 "${chain_sha}" \
  --canary "${base}/authority/ABF30B_DCP_EMBEDDING_CANARY_V1.json" --canary-sha256 "${canary_sha}" \
  --sequence-view "${sequence_view}" \
  --model-family "${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/protocols/antibody_model_family_v1.json" \
  --checkpoint "${root}/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1/segment-000069/run/checkpoint" \
  --esm2-root "${root}/data/sppider_seq_sabdab2_embeddings" --embedding-root "${embedding_root}" \
  --batch-antibodies 8 --resume
