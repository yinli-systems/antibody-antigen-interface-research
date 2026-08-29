#!/usr/bin/env bash
#SBATCH --job-name=fc-noneg-ensemble
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-noneg-ensemble-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-noneg-ensemble-%j.err

set -euo pipefail
umask 077

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/framecontact_20260730
readonly SOURCE_ROOT=${DATA_ROOT}/sabdab_eval_source_v1
readonly RUN_ROOT=${DATA_ROOT}/sabdab2/development-runs/negative-sample-ablation-v2/no_negative/framecontact
readonly OUTPUT_ROOT=${RUN_ROOT}/frozen-ensemble-v1

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]]
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 4090" ]]
[[ ! -e ${OUTPUT_ROOT} ]]

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=${SOURCE_ROOT}/src

exec "${PYTHON}" "${SOURCE_ROOT}/scripts/freeze_framecontact_development_ensemble.py" \
  --run-roots \
    "${RUN_ROOT}/seed-17" \
    "${RUN_ROOT}/seed-42" \
    "${RUN_ROOT}/seed-73" \
    "${RUN_ROOT}/seed-101" \
    "${RUN_ROOT}/seed-137" \
  --structure-cache-root "${DATA_ROOT}/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "${DATA_ROOT}/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "${DATA_ROOT}/split_matrix_v3_1/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --validation-partner-panel "${DATA_ROOT}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json" \
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \
  --panel-size 31 \
  --output-root "${OUTPUT_ROOT}" \
  --evaluation-batch-size 8 \
  --workers 4 \
  --device cuda \
  --precision bf16
