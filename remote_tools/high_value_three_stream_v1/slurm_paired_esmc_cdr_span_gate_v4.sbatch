#!/usr/bin/env bash
#SBATCH -J esmc-cdr-span-v4
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/cdr-span-v4-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/cdr-span-v4-%j.err

set -euo pipefail
: "${EXPECTED_AUTHORITY_SHA256:?submit with frozen authority SHA-256}"
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
test "${SLURM_JOB_PARTITION}" = gpu_4090
printf '%s  %s\n' "${EXPECTED_AUTHORITY_SHA256}" "${ROOT}/authority/PAIRED_ESMC_CDR_SPAN_GATE_AUTHORITY_V4.json" | sha256sum --check --strict
"${PYTHON}" "${SOURCE}/build_paired_esmc_cdr_span_gate_v4.py" \
  --authority "${ROOT}/authority/PAIRED_ESMC_CDR_SPAN_GATE_AUTHORITY_V4.json" \
  --source /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821/oas_stage_b_devclean_protocol_v3_unordered_v2.tsv.gz \
  --official-archive /ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/abcdr_oas_paired_18762309/OAS_paired_set.zip \
  --output-spans "${ROOT}/corpus/paired_esmc_cdr_spans_and_split_v4.tsv.gz" \
  --output-result "${ROOT}/PAIRED_ESMC_CDR_SPAN_GATE_RESULT_V4.json" \
  --output-receipt "${ROOT}/PAIRED_ESMC_CDR_SPAN_GATE_RECEIPT_V4.json"
