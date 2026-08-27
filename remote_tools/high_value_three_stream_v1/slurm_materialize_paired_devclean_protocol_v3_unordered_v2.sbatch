#!/usr/bin/env bash
#SBATCH -J esmc-cdr-filter-v3u2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/filter-v3u2-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/filter-v3u2-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly PLAN=${ROOT}/authority/PAIRED_DECONTAMINATION_EXECUTION_PLAN_V3.json
readonly PLAN_SHA256=${PAIRED_DECONTAMINATION_PLAN_SHA256:?missing plan SHA-256}
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "${SLURM_JOB_PARTITION}" == gpu_4090 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
"${PYTHON}" "${SOURCE}/materialize_paired_devclean_protocol_v3.py" \
 --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" \
 --merged-root "${ROOT}/merged-exclusions-protocol-v3-unordered-v2" \
 --output "${DATA}/oas_stage_b_devclean_protocol_v3_unordered_v2.tsv.gz" \
 --receipt "${DATA}/oas_stage_b_devclean_protocol_v3_unordered_v2.receipt.json"
