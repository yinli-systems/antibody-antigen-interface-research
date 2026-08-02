#!/bin/bash
#SBATCH --job-name=v6b-direct-highs
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/ssd/scxi253/model_deployment/logs/v6b-direct-highs-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/v6b-direct-highs-%j.err

set -euo pipefail

SOURCE_ROOT=${V6B_HIGHS_SOURCE_ROOT:?V6B_HIGHS_SOURCE_ROOT is required}
REGISTRATION=${V6B_HIGHS_REGISTRATION:?V6B_HIGHS_REGISTRATION is required}
REGISTRATION_SHA256=${V6B_HIGHS_REGISTRATION_SHA256:?V6B_HIGHS_REGISTRATION_SHA256 is required}
OUTPUT_DIRECTORY=${V6B_HIGHS_OUTPUT_DIRECTORY:?V6B_HIGHS_OUTPUT_DIRECTORY is required}
PYTHON_BIN=${V6B_HIGHS_PYTHON_BIN:?V6B_HIGHS_PYTHON_BIN is required}
CONTROLLER=$SOURCE_ROOT/scripts/run_v6_b_direct_highs_audit.py

umask 077
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH=$SOURCE_ROOT/src
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Registered HiGHS Python runtime is not executable" >&2
  exit 2
fi
if [[ ! -f "$CONTROLLER" || -L "$CONTROLLER" ]]; then
  echo "Direct-HiGHS controller is absent or symbolic" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIRECTORY" ]]; then
  echo "Direct-HiGHS output already exists" >&2
  exit 2
fi

/usr/bin/time -v "$PYTHON_BIN" "$CONTROLLER" \
  --registration "$REGISTRATION" \
  --registration-sha256 "$REGISTRATION_SHA256" \
  --output-directory "$OUTPUT_DIRECTORY"
