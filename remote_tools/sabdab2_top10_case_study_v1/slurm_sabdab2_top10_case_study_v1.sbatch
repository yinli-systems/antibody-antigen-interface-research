#!/usr/bin/env bash
#SBATCH --job-name=sabdab2-top10-case
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/sabdab2_top10_case_study_v1/logs/evaluate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/sabdab2_top10_case_study_v1/logs/evaluate-%j.err

set -euo pipefail
umask 077
: "${EXPECTED_PROTOCOL_SHA256:?submit with the frozen protocol SHA-256}"

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/sabdab2_top10_case_study_v1
readonly test_root=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly evaluator=${root}/tools/sabdab2_top10_case_study_v1/evaluate_sabdab2_top10_case_study_v1.py
readonly protocol=${campaign}/authority/PROTOCOL_SABDAB2_TOP10_CASE_STUDY_V1.json
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
test "$(sha256sum "${protocol}" | awk '{print $1}')" = "${EXPECTED_PROTOCOL_SHA256}"
mkdir -p "${campaign}/logs" "${campaign}/results"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${python_bin}" "${evaluator}" \
  --protocol "${protocol}" \
  --protocol-sha256 "${EXPECTED_PROTOCOL_SHA256}" \
  --structure-root "${test_root}/data/test_structure_cache_v1" \
  --contact-root "${test_root}/data/test_contact_authority_adapter_v1" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --output "${campaign}/results/TOP10_CASE_STUDY_RESULT.json" \
  --receipt "${campaign}/results/TOP10_CASE_STUDY_RECEIPT.json"
