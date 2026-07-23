#!/bin/bash
#SBATCH --job-name=antigen-folds
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/build_train_only_antigen_folds.py
AUTHORITY=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
CANONICAL_MANIFEST=$DEPLOY_ROOT/data/interaction_first/canonical_manifest.csv
EVIDENCE_ROOT=$DEPLOY_ROOT/data/interaction_first/identity80_development/exhaustive_mmseqs
ANTIGEN_EVIDENCE=$EVIDENCE_ROOT/antigen_identity80_exhaustive.tsv
ANTIBODY_EVIDENCE=$EVIDENCE_ROOT/antibody_identity80_exhaustive.tsv
OUTPUT_DIR=${ANTIGEN_FOLD_OUTPUT_DIR:-$DEPLOY_ROOT/data/interaction_first/antigen_cluster_development_folds}

NW_ARGS=()
if [[ -n "${ANTIGEN_NW_EDGES:-}" ]]; then
  NW_ARGS=(--needleman-wunsch-antigen-edges "$ANTIGEN_NW_EDGES")
fi

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" \
  --authority-split "$AUTHORITY" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --antigen-similarity-tsv "$ANTIGEN_EVIDENCE" \
  --antibody-similarity-tsv "$ANTIBODY_EVIDENCE" \
  "${NW_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --fold-count 3 \
  --identity-threshold 0.80 \
  --coverage-threshold 0.80 \
  --seed 42 \
  --resume
