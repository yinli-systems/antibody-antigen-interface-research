#!/bin/bash
#SBATCH --job-name=antigen-nw-audit
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/audit_train_only_antigen_folds.py
AUTHORITY=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
CANONICAL_MANIFEST=$DEPLOY_ROOT/data/interaction_first/canonical_manifest.csv
FOLD_ROOT=${ANTIGEN_FOLD_ROOT:-$DEPLOY_ROOT/data/interaction_first/antigen_cluster_development_folds}
FOLD_INDEX=$FOLD_ROOT/index.json
EXPECTED_INDEX_SHA256=${ANTIGEN_FOLD_INDEX_SHA256:?Set ANTIGEN_FOLD_INDEX_SHA256}
AUDIT_OUTPUT_DIR=${ANTIGEN_NW_AUDIT_OUTPUT_DIR:-$FOLD_ROOT/needleman_wunsch_audit}

PRIOR_ARGS=()
if [[ -n "${ANTIGEN_NW_PRIOR_EDGES:-}" ]]; then
  PRIOR_ARGS=(--prior-edge-tsv "$ANTIGEN_NW_PRIOR_EDGES")
fi

mkdir -p "$AUDIT_OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=1

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" \
  --authority-split "$AUTHORITY" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --fold-index "$FOLD_INDEX" \
  --expected-fold-index-sha256 "$EXPECTED_INDEX_SHA256" \
  "${PRIOR_ARGS[@]}" \
  --output-dir "$AUDIT_OUTPUT_DIR" \
  --identity-threshold 0.80 \
  --coverage-threshold 0.80 \
  --gap-open 10 \
  --gap-extend 1 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --chunk-size 512 \
  --top-pair-count 20 \
  --resume
