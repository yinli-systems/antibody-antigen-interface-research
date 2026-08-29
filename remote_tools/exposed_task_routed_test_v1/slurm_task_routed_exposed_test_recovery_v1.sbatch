#!/usr/bin/env bash
#SBATCH --job-name=task-routed-test-rec-v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/sabdab2_task_routed_seed17_historical_v1/logs/recovery-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/sabdab2_task_routed_seed17_historical_v1/logs/recovery-%A_%a.err

set -euo pipefail
umask 077
: "${EXPECTED_PROTOCOL_SHA256:?submit with the original frozen protocol SHA-256}"

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/sabdab2_task_routed_seed17_historical_v1
readonly data_root=${root}/data/framecontact_20260730
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly recovery=${root}/tools/recover_task_routed_exposed_test_component_v1.py

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    readonly arm=baseline_current
    readonly source_root=${root}/immutable_stage_a1_candidates/matched_local_training_source_v3
    ;;
  1)
    readonly arm=edge_topk2_supervised
    readonly source_root=${root}/immutable_stage_a1_candidates/edge_topk2_training_source_v1
    ;;
  *) exit 64 ;;
esac

readonly output=${campaign}/component-results/${arm}/COMPONENT_RESULT.json
readonly receipt=${campaign}/component-results/${arm}/COMPONENT_RECEIPT.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum ${campaign}/authority/PROTOCOL_TASK_ROUTED_EXPOSED_TEST_V1.json | awk '{print $1}')" = "${EXPECTED_PROTOCOL_SHA256}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${python_bin}" "${recovery}" \
  --campaign-root "${campaign}" \
  --protocol-sha256 "${EXPECTED_PROTOCOL_SHA256}" \
  --arm "${arm}" \
  --source-root "${source_root}" \
  --structure-cache-root "${data_root}/sabdab2/structure_cohort_v2" \
  --structure-manifest "${data_root}/sabdab2/structure_cohort_v2/STRUCTURE_CACHE_MANIFEST.json" \
  --contact-cache-root "${data_root}/sabdab2/contact_cohort_v2" \
  --contact-manifest "${data_root}/sabdab2/contact_cohort_v2/CONTACT_CACHE_MANIFEST.json" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --embedding-manifest "${root}/data/sppider_seq_sabdab2_embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json" \
  --split-manifest "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --test-panel "${data_root}/partner_metadata_antigen_fold_v3/test-reciprocal-31.json" \
  --output "${output}" \
  --receipt "${receipt}" \
  --evaluation-batch-size 8 \
  --workers 6
