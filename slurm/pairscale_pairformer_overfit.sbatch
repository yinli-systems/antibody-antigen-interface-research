#!/bin/bash
#SBATCH --job-name=pairscale-overfit32
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/tmp/pairscale-overfit32-%j.out
#SBATCH --error=/tmp/pairscale-overfit32-%j.err

set -euo pipefail

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${PAIRSCALE_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${PAIRSCALE_SOURCE_ROOT:?PAIRSCALE_SOURCE_ROOT is required}
CACHE_ROOT=${PAIRSCALE_CACHE_ROOT:?PAIRSCALE_CACHE_ROOT is required}
OUTPUT_ROOT=${PAIRSCALE_OUTPUT_ROOT:?PAIRSCALE_OUTPUT_ROOT is required}
EPOCHS=${PAIRSCALE_EPOCHS:-200}
LEARNING_RATE=${PAIRSCALE_LEARNING_RATE:-0.001}
POSITIVE_WEIGHT=${PAIRSCALE_POSITIVE_WEIGHT:-1024}
CONTACT_RANKING_WEIGHT=${PAIRSCALE_CONTACT_RANKING_WEIGHT:-1.0}
HARD_NEGATIVE_RATIO=${PAIRSCALE_HARD_NEGATIVE_RATIO:-8}
PARTNER_RANKING_WEIGHT=${PAIRSCALE_PARTNER_RANKING_WEIGHT:-0.25}
EVAL_EVERY=${PAIRSCALE_EVAL_EVERY:-10}
CHECKPOINT_EVERY=${PAIRSCALE_CHECKPOINT_EVERY:-10}
PRECISION=${PAIRSCALE_PRECISION:-bf16}
SEED=${PAIRSCALE_SEED:-17}

MANIFEST_SHA256=${PAIRSCALE_MANIFEST_SHA256:?PAIRSCALE_MANIFEST_SHA256 is required}
RECEIPT_SHA256=${PAIRSCALE_RECEIPT_FILE_SHA256:?PAIRSCALE_RECEIPT_FILE_SHA256 is required}
POLICY_SHA256=${PAIRSCALE_POLICY_SHA256:?PAIRSCALE_POLICY_SHA256 is required}
DECISION_SHA256=${PAIRSCALE_DECISION_SHA256:?PAIRSCALE_DECISION_SHA256 is required}
SUBSET_SHA256=${PAIRSCALE_SUBSET_MANIFEST_SHA256:?PAIRSCALE_SUBSET_MANIFEST_SHA256 is required}
AUTHORIZATION_SHA256=${PAIRSCALE_AUTHORIZATION_MANIFEST_SHA256:?PAIRSCALE_AUTHORIZATION_MANIFEST_SHA256 is required}

if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$EVAL_EVERY" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]] \
  || [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Epoch, interval, and seed settings must be canonical integers" >&2
  exit 2
fi
for digest in \
  "$MANIFEST_SHA256" \
  "$RECEIPT_SHA256" \
  "$POLICY_SHA256" \
  "$DECISION_SHA256" \
  "$SUBSET_SHA256" \
  "$AUTHORIZATION_SHA256"; do
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Every external data anchor must be a lowercase SHA-256 digest" >&2
    exit 2
  fi
done
if [[ "$PRECISION" != "fp32" && "$PRECISION" != "bf16" ]]; then
  echo "PAIRSCALE_PRECISION must be fp32 or bf16" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "The locked Python runtime is unavailable" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_ROOT/scripts/train_pairscale_pairformer_overfit.py" ]]; then
  echo "The sealed source root is incomplete" >&2
  exit 2
fi
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "The isolated cache root is unavailable" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "The output root is create-only" >&2
  exit 2
fi

export PYTHONPATH="$SOURCE_ROOT/src"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

RESUME_ARGUMENTS=()
if [[ -n "${PAIRSCALE_RESUME_CHECKPOINT_ROOT:-}" \
  || -n "${PAIRSCALE_RESUME_GENERATION:-}" \
  || -n "${PAIRSCALE_RESUME_RECEIPT_SHA256:-}" ]]; then
  if [[ -z "${PAIRSCALE_RESUME_CHECKPOINT_ROOT:-}" \
    || -z "${PAIRSCALE_RESUME_GENERATION:-}" \
    || ! "${PAIRSCALE_RESUME_RECEIPT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Resume root, generation, and receipt anchor are inseparable" >&2
    exit 2
  fi
  RESUME_ARGUMENTS=(
    --resume-checkpoint-root "$PAIRSCALE_RESUME_CHECKPOINT_ROOT"
    --resume-generation "$PAIRSCALE_RESUME_GENERATION"
    --resume-receipt-sha256 "$PAIRSCALE_RESUME_RECEIPT_SHA256"
  )
fi

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/train_pairscale_pairformer_overfit.py" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --manifest-sha256 "$MANIFEST_SHA256" \
  --receipt-file-sha256 "$RECEIPT_SHA256" \
  --policy-sha256 "$POLICY_SHA256" \
  --decision-sha256 "$DECISION_SHA256" \
  --subset-manifest-sha256 "$SUBSET_SHA256" \
  --authorization-manifest-sha256 "$AUTHORIZATION_SHA256" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --positive-weight "$POSITIVE_WEIGHT" \
  --contact-ranking-weight "$CONTACT_RANKING_WEIGHT" \
  --hard-negative-ratio "$HARD_NEGATIVE_RATIO" \
  --partner-ranking-weight "$PARTNER_RANKING_WEIGHT" \
  --eval-every "$EVAL_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --precision "$PRECISION" \
  --seed "$SEED" \
  "${RESUME_ARGUMENTS[@]}"
