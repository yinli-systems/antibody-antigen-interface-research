#!/bin/bash
#SBATCH --job-name=abcdresmc-bind
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/abcdresmc-download-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/abcdresmc-download-%j.err

set -euo pipefail

readonly python=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly root=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly source=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source

test "${SLURM_JOB_PARTITION}" = gpu_4090
nvidia-smi --query-gpu=name --format=csv,noheader | grep -Fx 'NVIDIA GeForce RTX 4090'
mkdir -p "${root}/logs"
exec "${python}" "${source}/download_verify_abcdresmc_checkpoint_v1.py" \
  --binding "${root}/authority/MODEL_BINDING_PAIRED_ESMC_CDR_V1.json" \
  --output-root "${root}"
