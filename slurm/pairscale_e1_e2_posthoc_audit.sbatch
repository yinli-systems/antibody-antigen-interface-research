#!/bin/bash
#SBATCH --job-name=pairscale-e1e2-audit
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pairscale-e1e2-audit-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pairscale-e1e2-audit-%j.err

set -euo pipefail

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${PAIRSCALE_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${PAIRSCALE_SOURCE_ROOT:?PAIRSCALE_SOURCE_ROOT is required}
CACHE_ROOT=${PAIRSCALE_CACHE_ROOT:?PAIRSCALE_CACHE_ROOT is required}
ASSIGNMENT_ROOT=${PAIRSCALE_ASSIGNMENT_ROOT:?PAIRSCALE_ASSIGNMENT_ROOT is required}
RUN_ROOT=${PAIRSCALE_RUN_ROOT:?PAIRSCALE_RUN_ROOT is required}
OUTPUT_ROOT=${PAIRSCALE_OUTPUT_ROOT:?PAIRSCALE_OUTPUT_ROOT is required}
SCREENING_SEED=${PAIRSCALE_SEED:?PAIRSCALE_SEED is required}

if [[ "$SCREENING_SEED" != 17 \
  && "$SCREENING_SEED" != 42 \
  && "$SCREENING_SEED" != 73 ]]; then
  echo "PAIRSCALE_SEED is not preregistered" >&2
  exit 2
fi

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
  PAIRSCALE_REPORT_SHA256
  PAIRSCALE_ORIGINAL_ACCESS_RECEIPT_SHA256
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
if [[ ! -f "$SOURCE_ROOT/scripts/audit_pairscale_e1_e2_checkpoint.py" ]]; then
  echo "The sealed posthoc source root is incomplete" >&2
  exit 2
fi
observed_controller_sha256=$(
  sha256sum "$SOURCE_ROOT/scripts/audit_pairscale_e1_e2_checkpoint.py" \
    | cut -d' ' -f1
)
if [[ "$observed_controller_sha256" != "$PAIRSCALE_CONTROLLER_SHA256" ]]; then
  echo "Posthoc controller source digest differs" >&2
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
if [[ ! -f "$RUN_ROOT/FINAL_REPORT.json" \
  || ! -f "$RUN_ROOT/DEVELOPMENT_HOLDOUT_ACCESS_RECEIPT.json" ]]; then
  echo "The completed source run is unavailable" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "The posthoc output root is create-only" >&2
  exit 2
fi

export PYTHONPATH="$SOURCE_ROOT/src"
export PYTHONHASHSEED="$SCREENING_SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/audit_pairscale_e1_e2_checkpoint.py" \
  --seed "$SCREENING_SEED" \
  --cache-root "$CACHE_ROOT" \
  --manifest-sha256 "$PAIRSCALE_MANIFEST_SHA256" \
  --cache-receipt-file-sha256 "$PAIRSCALE_CACHE_RECEIPT_FILE_SHA256" \
  --policy-sha256 "$PAIRSCALE_POLICY_SHA256" \
  --decision-sha256 "$PAIRSCALE_DECISION_SHA256" \
  --subset-manifest-sha256 "$PAIRSCALE_SUBSET_MANIFEST_SHA256" \
  --authorization-manifest-sha256 \
    "$PAIRSCALE_AUTHORIZATION_MANIFEST_SHA256" \
  --assignment "$ASSIGNMENT_ROOT/DEVELOPMENT_COMPONENT_HOLDOUT.json" \
  --assignment-file-sha256 "$PAIRSCALE_ASSIGNMENT_FILE_SHA256" \
  --assignment-content-sha256 "$PAIRSCALE_ASSIGNMENT_CONTENT_SHA256" \
  --assignment-receipt "$ASSIGNMENT_ROOT/HOLDOUT_BUILD_RECEIPT.json" \
  --assignment-receipt-file-sha256 \
    "$PAIRSCALE_ASSIGNMENT_RECEIPT_FILE_SHA256" \
  --report "$RUN_ROOT/FINAL_REPORT.json" \
  --report-sha256 "$PAIRSCALE_REPORT_SHA256" \
  --original-access-receipt \
    "$RUN_ROOT/DEVELOPMENT_HOLDOUT_ACCESS_RECEIPT.json" \
  --original-access-receipt-sha256 \
    "$PAIRSCALE_ORIGINAL_ACCESS_RECEIPT_SHA256" \
  --checkpoint-root "$RUN_ROOT/checkpoints" \
  --output-root "$OUTPUT_ROOT"
