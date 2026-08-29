#!/usr/bin/env bash
#SBATCH --job-name=topk2-multitask-readout
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/sabdab2_multitask_topk2_readout_v1/logs/evaluate-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/sabdab2_multitask_topk2_readout_v1/logs/evaluate-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tool_root=${root}/tools/multitask_topk2_readout_v1
readonly result_parent=${root}/results/development/sabdab2_multitask_topk2_readout_v1
readonly output=${result_parent}/frozen_result
readonly source_root=${root}/immutable_stage_a1_candidates/epitope_pooling_ablation_source_v1
readonly source_manifest=${source_root}/SOURCE_MANIFEST_EPITOPE_POOLING_ABLATION_V1.json
readonly run_root=${root}/results/development/sabdab2_epitope_pooling_ablation_v1/runs/topk2/seed-17
readonly data_root=${root}/data/framecontact_20260730
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
test ! -e "${output}"
mkdir -p "${result_parent}/logs"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

"${python_bin}" "${tool_root}/evaluate_multitask_topk2_readout_v1.py" \
  --protocol "${tool_root}/PROTOCOL_MULTITASK_TOPK2_READOUT_V1.json" \
  --protocol-receipt "${tool_root}/PROTOCOL_MULTITASK_TOPK2_READOUT_V1_RECEIPT.json" \
  --source-root "${source_root}" --source-manifest "${source_manifest}" \
  --run-result "${run_root}/FRAMECONTACT_DEVELOPMENT_RESULT.json" \
  --checkpoint "${run_root}/FRAMECONTACT_SELECTED.pt" \
  --structure-root "${data_root}/sabdab2/structure_cohort_v2" \
  --contact-root "${data_root}/sabdab2/contact_cohort_v2" \
  --embedding-root "${root}/data/sppider_seq_sabdab2_embeddings" \
  --split "${data_root}/split_matrix_v3_1/antigen_fold.csv" \
  --output-root "${output}" --workers 4 --device cuda
