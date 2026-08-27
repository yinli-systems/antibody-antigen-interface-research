#!/usr/bin/env bash
#SBATCH -J fc-temporal-double-cold-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/label-free-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/label-free-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1
readonly IDENTITY=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/inputs/SABDAB2_TEMPORAL_HOLDOUT_IDENTITY_MANIFEST_V1.json
readonly REFERENCES=/ssd/scxi253/model_deployment/data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1/sabdab2_train_validation_sequences_v1.jsonl
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_TEMPORAL_DOUBLE_COLD_V1.json

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${SLURM_JOB_PARTITION}" == "gpu_4090" && "${gpu_count}" -eq 1 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
protocol_sha=$(${PYTHON} -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${PROTOCOL}")
mkdir -p "${ROOT}/logs"
${PYTHON} "${SOURCE}/prepare_temporal_double_cold_v1.py" \
  --identity-manifest "${IDENTITY}" \
  --identity-manifest-sha256 23e6b9848216cdc57ea3210cba89905a659817b6a49a3de9815d9b5153da5303 \
  --reference-sequences "${REFERENCES}" \
  --reference-sequences-sha256 27e0beb755c3d8e4b9910d4ad2ca979161668a1e1fe7b4d09bf6a60251e07468 \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${protocol_sha}" \
  --output-root "${ROOT}" \
  --workers 6
