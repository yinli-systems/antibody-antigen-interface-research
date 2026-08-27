#!/usr/bin/env bash
#SBATCH -J esmc-cdr-hit-shards-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/sharding-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/sharding-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly DATA=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821
readonly OUTPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821/pairwise_hits_shards_v1
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_PAIRED_ESMC_CDR_V1.json

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${SLURM_JOB_PARTITION}" == "gpu_4090" && "${gpu_count}" -eq 1 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
protocol_sha=$(${PYTHON} -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${PROTOCOL}")
mkdir -p "${ROOT}/logs"
${PYTHON} "${SOURCE}/shard_paired_decontamination_hits_v1.py" \
  --hits "${DATA}/oas_stage_b_development_homology_hits_reversed_v3.tsv.gz" \
  --hits-sha256 edf4c952061ffef3e48b24ddb61a30d303a9cca26a3fa47c5ddcb6ce8f8f8338 \
  --source-receipt "${DATA}/oas_stage_b_exact_pairs_v2.receipt.json" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${protocol_sha}" \
  --output-root "${OUTPUT}" \
  --shards 64
