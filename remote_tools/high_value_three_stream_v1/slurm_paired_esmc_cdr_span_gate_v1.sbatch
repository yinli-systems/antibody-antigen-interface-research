#!/usr/bin/env bash
#SBATCH -J esmc-cdr-span
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/cdr-span-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/cdr-span-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
"${PYTHON}" "${SOURCE}/build_paired_esmc_cdr_span_gate_v1.py" \
  --authority "${ROOT}/authority/PAIRED_ESMC_CDR_SPAN_GATE_AUTHORITY_V1.json" \
  --source /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821/oas_stage_b_devclean_protocol_v3_unordered_v2.tsv.gz \
  --output-spans "${ROOT}/corpus/paired_esmc_cdr_spans_and_split_v1.tsv.gz" \
  --output-result "${ROOT}/PAIRED_ESMC_CDR_SPAN_GATE_RESULT_V1.json" \
  --output-receipt "${ROOT}/PAIRED_ESMC_CDR_SPAN_GATE_RECEIPT_V1.json"
