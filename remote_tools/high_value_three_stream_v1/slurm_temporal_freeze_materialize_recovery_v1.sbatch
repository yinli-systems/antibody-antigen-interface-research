#!/usr/bin/env bash
#SBATCH -J fc-temporal-open-recover-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/open-recover-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/open-recover-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1
readonly PLAN=${ROOT}/authority/TEMPORAL_LABEL_FREE_RECOVERY_PLAN_V3.json
readonly PLAN_SHA256=${TEMPORAL_LABEL_FREE_RECOVERY_PLAN_SHA256:?missing frozen recovery plan SHA-256}

"${PYTHON}" "${SOURCE}/verify_temporal_label_free_recovery_v1.py" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" \
  --manifest "${ROOT}/TEMPORAL_DOUBLE_COLD_LABEL_FREE_MANIFEST_V1.json" \
  --base-receipt "${ROOT}/TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECEIPT_V1.json" \
  --recovery-receipt "${ROOT}/TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECOVERY_RECEIPT_V1.json"

# The frozen post-gate freezer and materializer remain byte-for-byte unchanged.
bash "${SOURCE}/slurm_temporal_freeze_materialize_v1.sbatch"
