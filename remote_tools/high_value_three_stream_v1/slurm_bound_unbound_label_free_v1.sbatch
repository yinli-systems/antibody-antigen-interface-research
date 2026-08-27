#!/usr/bin/env bash
#SBATCH -J fc-bound-unbound-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/label-free-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1/logs/label-free-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/bound-unbound-geometry-v1
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_BOUND_UNBOUND_GEOMETRY_V1.json

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${SLURM_JOB_PARTITION}" == "gpu_4090" && "${gpu_count}" -eq 1 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
protocol_sha=$(${PYTHON} -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${PROTOCOL}")
mkdir -p "${ROOT}/logs"
${PYTHON} "${SOURCE}/prepare_bound_unbound_geometry_v1.py" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${protocol_sha}" \
  --output-root "${ROOT}"
