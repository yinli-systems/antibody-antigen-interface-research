#!/bin/bash
#SBATCH --job-name=v6b-direct-cp-audit
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/ssd/scxi253/model_deployment/logs/v6b-direct-cp-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/v6b-direct-cp-audit-%j.err

set -euo pipefail

CONTROLLER_ROOT=${V6B_AUDIT_CONTROLLER_ROOT:?V6B_AUDIT_CONTROLLER_ROOT is required}
MODULE_ROOT=${V6B_MODULE_ROOT:?V6B_MODULE_ROOT is required}
REGISTRATION=${V6B_REGISTRATION:?V6B_REGISTRATION is required}
REGISTRATION_SHA256=${V6B_REGISTRATION_SHA256:?V6B_REGISTRATION_SHA256 is required}
PRIMARY_RESULT=${V6B_PRIMARY_RESULT:?V6B_PRIMARY_RESULT is required}
PRIMARY_RESULT_SHA256=${V6B_PRIMARY_RESULT_SHA256:?V6B_PRIMARY_RESULT_SHA256 is required}
CONTROLLER_SHA256=${V6B_AUDIT_CONTROLLER_SHA256:?V6B_AUDIT_CONTROLLER_SHA256 is required}
OUTPUT_DIRECTORY=${V6B_AUDIT_OUTPUT_DIRECTORY:?V6B_AUDIT_OUTPUT_DIRECTORY is required}
PYTHON_BIN=${V6B_PYTHON_BIN:?V6B_PYTHON_BIN is required}
CONTROLLER=$CONTROLLER_ROOT/scripts/run_v6_b_independent_cp_audit.py

umask 077
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH=$CONTROLLER_ROOT/src:$MODULE_ROOT/src
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Registered Python runtime is not executable" >&2
  exit 2
fi
if [[ ! -f "$CONTROLLER" || -L "$CONTROLLER" ]]; then
  echo "Independent CP controller is absent or symbolic" >&2
  exit 2
fi
if [[ "$(sha256sum "$CONTROLLER" | cut -d ' ' -f 1)" != "$CONTROLLER_SHA256" ]]; then
  echo "Independent CP controller hash differs" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIRECTORY" ]]; then
  echo "Independent CP audit output already exists" >&2
  exit 2
fi

/usr/bin/time -v "$PYTHON_BIN" "$CONTROLLER" \
  --registration "$REGISTRATION" \
  --registration-sha256 "$REGISTRATION_SHA256" \
  --primary-result "$PRIMARY_RESULT" \
  --primary-result-sha256 "$PRIMARY_RESULT_SHA256" \
  --controller-sha256 "$CONTROLLER_SHA256" \
  --output-directory "$OUTPUT_DIRECTORY"
