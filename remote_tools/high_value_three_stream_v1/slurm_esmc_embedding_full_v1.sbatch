#!/usr/bin/env bash
#SBATCH -J esmc-embed-full
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/embed-full-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/embed-full-%A_%a.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly PLAN=${ROOT}/authority/SABDAB2_ESMC_DEVELOPMENT_SEQUENCE_PLAN_V1.json
readonly PLAN_SHA256=4bdd000eea6f4d4c514f995849f084a39b845f8dd94719eb7f667ab4db1ef0b2
readonly SHARDS=4
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "${SLURM_JOB_PARTITION}" == gpu_4090 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
if (( SLURM_ARRAY_TASK_ID < SHARDS )); then
  role=base
  shard=${SLURM_ARRAY_TASK_ID}
  checkpoint=${ROOT}/models/biohub--ESMC-600M/a7e82012c83126b9eedb055fea9fa84b6c02f094/model.safetensors
  expected=e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8
  output_root=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2_esmc600m_base_dev_v1
else
  role=published_cdr
  shard=$((SLURM_ARRAY_TASK_ID - SHARDS))
  checkpoint=${ROOT}/models/NOC-Lab--AbCDR-ESMC/d6e08c7a771f6c7b399e9f94cd3b9b36d0ac12b8/model.safetensors
  expected=0490932ca515c6fe1738d070ede8b3ade28745d990a5e1325f286f3624c1d4e6
  output_root=/ssd/scxi253/model_deployment/data/framecontact_20260730/sabdab2_abcdr_esmc_dev_v1
fi
"${PYTHON}" "${SOURCE}/materialize_esmc_embeddings_v1.py" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" --role "${role}" \
  --checkpoint "${checkpoint}" --checkpoint-sha256 "${expected}" \
  --safetensors-site /ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages \
  --output-root "${output_root}" \
  --result "${ROOT}/embedding-full-v1/${role}/shard-${shard}.json" \
  --shard-index "${shard}" --shard-count "${SHARDS}"
