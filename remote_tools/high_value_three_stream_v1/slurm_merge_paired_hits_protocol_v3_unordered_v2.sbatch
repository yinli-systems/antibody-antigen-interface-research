#!/usr/bin/env bash
#SBATCH -J esmc-cdr-merge-v3u2
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/merge-v3u2-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/merge-v3u2-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly PLAN=${ROOT}/authority/PAIRED_DECONTAMINATION_EXECUTION_PLAN_V3.json
readonly PLAN_SHA256=${PAIRED_DECONTAMINATION_PLAN_SHA256:?missing plan SHA-256}
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "${SLURM_JOB_PARTITION}" == gpu_4090 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
"${PYTHON}" "${SOURCE}/merge_paired_hit_shards_protocol_v3.py" \
 --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" \
 --reductions-root "${ROOT}/shard-reductions-protocol-v3-unordered-v2" \
 --output-root "${ROOT}/merged-exclusions-protocol-v3-unordered-v2"
