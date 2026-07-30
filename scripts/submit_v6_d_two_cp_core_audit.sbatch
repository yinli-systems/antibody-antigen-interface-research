#!/bin/bash
#SBATCH --job-name=v6d-two-cp-core
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/ssd/scxi253/model_deployment/logs/v6d-two-cp-core-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/v6d-two-cp-core-%j.err

set -euo pipefail

SOURCE_ROOT=${V6D_SOURCE_ROOT:?V6D_SOURCE_ROOT is required}
REGISTRATION=${V6D_REGISTRATION:?V6D_REGISTRATION is required}
REGISTRATION_SHA256=${V6D_REGISTRATION_SHA256:?V6D_REGISTRATION_SHA256 is required}
CONTROLLER_SHA256=${V6D_CONTROLLER_SHA256:?V6D_CONTROLLER_SHA256 is required}
OUTPUT_DIRECTORY=${V6D_OUTPUT_DIRECTORY:?V6D_OUTPUT_DIRECTORY is required}
PYTHON_BIN=${V6D_PYTHON_BIN:?V6D_PYTHON_BIN is required}
CONTROLLER=$SOURCE_ROOT/scripts/run_v6_d_two_cp_core_audit.py

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
  echo "Registered V6-D Python runtime is not executable" >&2
  exit 2
fi
if [[ ! -f "$CONTROLLER" || -L "$CONTROLLER" ]]; then
  echo "V6-D controller is absent or symbolic" >&2
  exit 2
fi
if [[ "$(sha256sum "$CONTROLLER" | cut -d ' ' -f 1)" != "$CONTROLLER_SHA256" ]]; then
  echo "V6-D controller hash differs" >&2
  exit 2
fi
if [[ ! -f "$REGISTRATION" || -L "$REGISTRATION" ]]; then
  echo "V6-D registration is absent or symbolic" >&2
  exit 2
fi
if [[ "$(sha256sum "$REGISTRATION" | cut -d ' ' -f 1)" != "$REGISTRATION_SHA256" ]]; then
  echo "V6-D registration hash differs" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIRECTORY" || -L "$OUTPUT_DIRECTORY" ]]; then
  echo "V6-D output already exists" >&2
  exit 2
fi

/usr/bin/time -v "$PYTHON_BIN" "$CONTROLLER" \
  --registration "$REGISTRATION" \
  --registration-sha256 "$REGISTRATION_SHA256" \
  --controller-sha256 "$CONTROLLER_SHA256" \
  --output-directory "$OUTPUT_DIRECTORY"
