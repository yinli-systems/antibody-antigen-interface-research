#!/usr/bin/env bash
#SBATCH --job-name=pairscale-materialize
#SBATCH --partition=vip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/%x-%j.err

set -euo pipefail

: "${PAIR_SOURCE_ROOT:?PAIR_SOURCE_ROOT is required}"
: "${PAIR_SOURCE_ID:?PAIR_SOURCE_ID is required}"
: "${PAIR_DESTINATION_ROOT:?PAIR_DESTINATION_ROOT is required}"

export PYTHONPATH="${PAIR_SOURCE_ROOT}/src"
mkdir -p /ssd/scxi253/model_deployment/logs

exec python3 "${PAIR_SOURCE_ROOT}/scripts/materialize_registered_source.py" \
  --plan "${PAIR_SOURCE_ROOT}/data_registry/download_plans/${PAIR_SOURCE_ID}.json" \
  --registry-root "${PAIR_SOURCE_ROOT}/data_registry/sources" \
  --destination-root "${PAIR_DESTINATION_ROOT}" \
  --workers "${PAIR_SOURCE_WORKERS:-8}"
