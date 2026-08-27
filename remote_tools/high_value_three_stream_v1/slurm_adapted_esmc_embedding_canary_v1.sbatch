#!/usr/bin/env bash
#SBATCH -J esmc-adapt-embed-canary
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-canary-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-canary-%A_%a.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly PLAN=${ROOT}/authority/SABDAB2_ESMC_DEVELOPMENT_SEQUENCE_PLAN_V1.json
readonly PLAN_SHA256=4bdd000eea6f4d4c514f995849f084a39b845f8dd94719eb7f667ab4db1ef0b2
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_ADAPTED_ESMC_EMBEDDING_DEVELOPMENT_V1.json
PROTOCOL_SHA256=$(sha256sum "${PROTOCOL}" | awk '{print $1}')
readonly PROTOCOL_SHA256
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "${SLURM_JOB_PARTITION}" == gpu_4090 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
if (( SLURM_ARRAY_TASK_ID == 0 )); then
  role=whole_chain_adapted
  checkpoint=${ROOT}/adaptation-production-v1/whole_chain_control/checkpoint-epoch-4.safetensors
  expected=50b27675df33f843e7a9a71eca4423c449603246980e3296dfbb21fc27ffa1fb
else
  role=cdr_focused_adapted
  checkpoint=${ROOT}/adaptation-production-v1/cdr_focused/checkpoint-epoch-4.safetensors
  expected=5fd67e390f4602f88e9a59d03a11bff332da487ebe4425bdb562bcce8a3a9d9a
fi
"${PYTHON}" "${SOURCE}/materialize_adapted_esmc_embeddings_v1.py" \
  --protocol "${PROTOCOL}" --protocol-sha256 "${PROTOCOL_SHA256}" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" --role "${role}" \
  --checkpoint "${checkpoint}" --checkpoint-sha256 "${expected}" \
  --safetensors-site /ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages \
  --output-root "${ROOT}/adapted-embedding-canary-v1/${role}" \
  --result "${ROOT}/adapted-embedding-canary-v1/${role}.json" --canary
