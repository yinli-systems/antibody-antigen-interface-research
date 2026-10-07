#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-freeze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/freeze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/freeze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly base=${root}/results/development/abf30b_framecontact_v1
readonly protocol=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json
readonly receipt=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha=$("${python_bin}" - "${protocol}" "${receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); r=json.load(open(sys.argv[2])); d=hashlib.sha256(raw).hexdigest(); assert r['protocol_sha256']==d; print(d)
PY
)
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools}/freeze_test_measurement_v1.py" \
  --campaign-root "${campaign}" --prep-protocol "${protocol}" --prep-protocol-sha256 "${protocol_sha}" \
  --final-development-root "${base}" \
  --baseline-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_baseline_source_v1" \
  --topk2-source-root "${root}/immutable_stage_a1_candidates/abf30b_embedding_topk2_source_v1" \
  --structure-root "${campaign}/data/test_structure_cache_v1" \
  --embedding-root "${campaign}/embeddings/sabdab2_test_abf30b_antibody_esm2_antigen_v1" \
  --contact-manifest "${root}/data/framecontact_20260730/sabdab2/contact_cohort_v2/CONTACT_CACHE_MANIFEST.json" \
  --split "${root}/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv" \
  --test-panel "${root}/data/framecontact_20260730/partner_metadata_antigen_fold_v3/test-reciprocal-31.json" \
  --evaluator "${tools}/evaluate_exposed_test_component_v1.py" \
  --analyzer "${tools}/analyze_exposed_test_v1.py" --sbatch "${tools}/evaluate_exposed_test_v1.sbatch" \
  --output "${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json" \
  --receipt-output "${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1_RECEIPT.json"
