#!/usr/bin/env bash
#SBATCH --job-name=abf30b-exposed-test
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/eval-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/eval-%A_%a.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly execution=${root}/results/development/abf30b_framecontact_v1/authority/development_execution_v1
readonly protocol=${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json
readonly protocol_receipt=${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10
case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly arm=baseline; readonly seed=17 ;;
  1) readonly arm=baseline; readonly seed=29 ;;
  2) readonly arm=baseline; readonly seed=43 ;;
  3) readonly arm=edge_topk2_supervised; readonly seed=17 ;;
  4) readonly arm=edge_topk2_supervised; readonly seed=29 ;;
  5) readonly arm=edge_topk2_supervised; readonly seed=43 ;;
  *) exit 64 ;;
esac
if test "${arm}" = baseline; then
  readonly source_root=${root}/immutable_stage_a1_candidates/abf30b_embedding_baseline_source_v1
  readonly source_manifest=${execution}/SOURCE_MANIFEST_BASELINE_ABF30B_V1.json
else
  readonly source_root=${root}/immutable_stage_a1_candidates/abf30b_embedding_topk2_source_v1
  readonly source_manifest=${execution}/SOURCE_MANIFEST_EDGE_TOPK2_SUPERVISED_ABF30B_V1.json
fi
readonly protocol_sha=$("${python_bin}" - "${protocol}" "${protocol_receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); r=json.load(open(sys.argv[2])); d=hashlib.sha256(raw).hexdigest(); assert r['protocol_sha256']==d; print(d)
PY
)
readonly output=${campaign}/component_results/${arm}/seed-${seed}/COMPONENT_RESULT.json
readonly receipt=${campaign}/component_results/${arm}/seed-${seed}/COMPONENT_RECEIPT.json
test "${SLURM_JOB_PARTITION}" = gpu_4090
test -n "${CUDA_VISIBLE_DEVICES:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${source_root}/src
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" "${tools}/evaluate_exposed_test_component_v1.py" \
  --campaign-root "${campaign}" --protocol-sha256 "${protocol_sha}" \
  --arm "${arm}" --seed "${seed}" --source-root "${source_root}" --source-manifest "${source_manifest}" \
  --structure-root "${campaign}/data/test_structure_cache_v1" \
  --contact-root "${root}/data/framecontact_20260730/sabdab2/contact_cohort_v2" \
  --embedding-root "${campaign}/embeddings/sabdab2_test_abf30b_antibody_esm2_antigen_v1" \
  --split "${root}/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv" \
  --test-panel "${root}/data/framecontact_20260730/partner_metadata_antigen_fold_v3/test-reciprocal-31.json" \
  --output "${output}" --receipt "${receipt}"
