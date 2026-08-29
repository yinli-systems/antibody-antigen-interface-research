#!/usr/bin/env bash
#SBATCH -J esmc-embed-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/embed-audit-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/embed-audit-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
"${PYTHON}" "${SOURCE}/finalize_esmc_embeddings_v1.py" \
  --plan "${ROOT}/authority/SABDAB2_ESMC_DEVELOPMENT_SEQUENCE_PLAN_V1.json" \
  --plan-sha256 4bdd000eea6f4d4c514f995849f084a39b845f8dd94719eb7f667ab4db1ef0b2 \
  --canary-receipt "${ROOT}/ESMC_EMBEDDING_CANARY_ANALYSIS_V1_RECEIPT.json" \
  --result-root "${ROOT}/embedding-full-v1" \
  --base-root /ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2_esmc600m_base_dev_v1 \
  --cdr-root /ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2_abcdr_esmc_dev_v1 \
  --output "${ROOT}/ESMC_EMBEDDING_FULL_ANALYSIS_V1.json" \
  --receipt "${ROOT}/ESMC_EMBEDDING_FULL_ANALYSIS_V1_RECEIPT.json"
