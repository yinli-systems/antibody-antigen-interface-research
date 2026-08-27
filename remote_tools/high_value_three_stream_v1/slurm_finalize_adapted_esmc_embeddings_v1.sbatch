#!/usr/bin/env bash
#SBATCH -J esmc-adapt-embed-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-audit-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-audit-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly DATA=/ssd/scxi253/model_deployment/data/framecontact_20260730
"${PYTHON}" "${SOURCE}/finalize_adapted_esmc_embeddings_v1.py" \
  --execution "${ROOT}/authority/EXECUTION_ADAPTED_ESMC_EMBEDDING_FULL_V1.json" \
  --execution-sha256 aa394f541283602ad17ac9db7771fad5aa163df258ed753cafdfc6585c9c0ac3 \
  --plan "${ROOT}/authority/SABDAB2_ESMC_DEVELOPMENT_SEQUENCE_PLAN_V1.json" \
  --base-root "${DATA}/sabdab2_esmc600m_base_dev_v1" \
  --base-manifest "${DATA}/sabdab2_esmc600m_base_dev_v1/SABDAB2_ESM_EMBEDDING_MANIFEST.json" \
  --result-root "${ROOT}/adapted-embedding-full-v1" \
  --whole-root "${DATA}/sabdab2_esmc600m_whole_chain_adapted_dev_v1" \
  --cdr-root "${DATA}/sabdab2_esmc600m_cdr_focused_adapted_dev_v1" \
  --output "${ROOT}/ADAPTED_ESMC_EMBEDDING_FULL_ANALYSIS_V1.json" \
  --receipt "${ROOT}/ADAPTED_ESMC_EMBEDDING_FULL_ANALYSIS_V1_RECEIPT.json"
