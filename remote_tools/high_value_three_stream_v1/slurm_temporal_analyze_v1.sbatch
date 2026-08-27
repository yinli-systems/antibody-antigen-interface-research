#!/usr/bin/env bash
#SBATCH -J fc-temporal-audit-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/analyze-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/analyze-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_TEMPORAL_DOUBLE_COLD_MEASUREMENT_V1.json
readonly DATA=${ROOT}/data-v1
sha() { "${PYTHON}" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
protocol_sha=$(sha "${PROTOCOL}")
data_sha=$(sha "${DATA}/TEMPORAL_DOUBLE_COLD_DATA_MANIFEST_V1.json")
"${PYTHON}" "${SOURCE}/analyze_temporal_double_cold_v1.py" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${protocol_sha}" \
  --data-root "${DATA}" \
  --data-manifest-sha256 "${data_sha}" \
  --components-root "${ROOT}/components-v1" \
  --output "${ROOT}/TEMPORAL_DOUBLE_COLD_MEASUREMENT_RESULT_V1.json" \
  --receipt "${ROOT}/TEMPORAL_DOUBLE_COLD_MEASUREMENT_RESULT_V1_RECEIPT.json"
