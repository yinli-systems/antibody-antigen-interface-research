#!/bin/bash
#SBATCH --job-name=pairscale-multidecoy260
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pairscale-multidecoy260-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pairscale-multidecoy260-%j.err

set -euo pipefail

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${PAIRSCALE_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${PAIRSCALE_SOURCE_ROOT:?PAIRSCALE_SOURCE_ROOT is required}
CACHE_ROOT=${PAIRSCALE_CACHE_ROOT:?PAIRSCALE_CACHE_ROOT is required}
ASSIGNMENT_ROOT=${PAIRSCALE_ASSIGNMENT_ROOT:?PAIRSCALE_ASSIGNMENT_ROOT is required}
OUTPUT_ROOT=${PAIRSCALE_OUTPUT_ROOT:?PAIRSCALE_OUTPUT_ROOT is required}

required_digests=(
  PAIRSCALE_MANIFEST_SHA256
  PAIRSCALE_CACHE_RECEIPT_FILE_SHA256
  PAIRSCALE_POLICY_SHA256
  PAIRSCALE_DECISION_SHA256
  PAIRSCALE_SUBSET_MANIFEST_SHA256
  PAIRSCALE_AUTHORIZATION_MANIFEST_SHA256
  PAIRSCALE_ASSIGNMENT_FILE_SHA256
  PAIRSCALE_ASSIGNMENT_CONTENT_SHA256
  PAIRSCALE_ASSIGNMENT_RECEIPT_FILE_SHA256
  PAIRSCALE_PROTOCOL_FILE_SHA256
  PAIRSCALE_CONTROLLER_SHA256
  PAIRSCALE_SCHEDULER_SCRIPT_SHA256
)
for name in "${required_digests[@]}"; do
  value=${!name:-}
  if [[ ! "$value" =~ ^[0-9a-f]{64}$ ]]; then
    echo "$name must be a lowercase SHA-256" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "The locked Python runtime is unavailable" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ROOT/scripts/train_pairscale_centered_multidecoy_repair.py" ]]; then
  echo "The sealed centered multi-decoy source root is incomplete" >&2
  exit 2
fi
observed_scheduler_sha256=$(sha256sum "$0" | cut -d' ' -f1)
if [[ "$observed_scheduler_sha256" != "$PAIRSCALE_SCHEDULER_SCRIPT_SHA256" ]]; then
  echo "Scheduler script digest differs" >&2
  exit 2
fi
if [[ ! -d "$CACHE_ROOT" || ! -d "$ASSIGNMENT_ROOT" ]]; then
  echo "The cache or assignment root is unavailable" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "The output root is create-only" >&2
  exit 2
fi

export PYTHONPATH="$SOURCE_ROOT/src"
export PYTHONHASHSEED=17
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

resume_arguments=()
if [[ -n "${PAIRSCALE_RESUME_CHECKPOINT_ROOT:-}" \
  || -n "${PAIRSCALE_RESUME_GENERATION:-}" \
  || -n "${PAIRSCALE_RESUME_RECEIPT_SHA256:-}" ]]; then
  if [[ -z "${PAIRSCALE_RESUME_CHECKPOINT_ROOT:-}" \
    || -z "${PAIRSCALE_RESUME_GENERATION:-}" \
    || ! "${PAIRSCALE_RESUME_RECEIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Resume root, generation, and receipt are inseparable" >&2
    exit 2
  fi
  resume_arguments=(
    --resume-checkpoint-root "$PAIRSCALE_RESUME_CHECKPOINT_ROOT"
    --resume-generation "$PAIRSCALE_RESUME_GENERATION"
    --resume-receipt-sha256 "$PAIRSCALE_RESUME_RECEIPT_SHA256"
  )
fi

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/train_pairscale_centered_multidecoy_repair.py" \
  --cache-root "$CACHE_ROOT" \
  --manifest-sha256 "$PAIRSCALE_MANIFEST_SHA256" \
  --cache-receipt-file-sha256 "$PAIRSCALE_CACHE_RECEIPT_FILE_SHA256" \
  --policy-sha256 "$PAIRSCALE_POLICY_SHA256" \
  --decision-sha256 "$PAIRSCALE_DECISION_SHA256" \
  --subset-manifest-sha256 "$PAIRSCALE_SUBSET_MANIFEST_SHA256" \
  --authorization-manifest-sha256 "$PAIRSCALE_AUTHORIZATION_MANIFEST_SHA256" \
  --assignment "$ASSIGNMENT_ROOT/DEVELOPMENT_COMPONENT_HOLDOUT.json" \
  --assignment-file-sha256 "$PAIRSCALE_ASSIGNMENT_FILE_SHA256" \
  --assignment-content-sha256 "$PAIRSCALE_ASSIGNMENT_CONTENT_SHA256" \
  --assignment-receipt "$ASSIGNMENT_ROOT/HOLDOUT_BUILD_RECEIPT.json" \
  --assignment-receipt-file-sha256 \
    "$PAIRSCALE_ASSIGNMENT_RECEIPT_FILE_SHA256" \
  --protocol "$SOURCE_ROOT/protocols/pairscale_centered_multidecoy_repair_v1.json" \
  --protocol-file-sha256 "$PAIRSCALE_PROTOCOL_FILE_SHA256" \
  --expected-controller-sha256 "$PAIRSCALE_CONTROLLER_SHA256" \
  --scheduler-script-sha256 "$PAIRSCALE_SCHEDULER_SCRIPT_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  "${resume_arguments[@]}"
