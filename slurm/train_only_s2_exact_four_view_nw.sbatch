#!/bin/bash
#SBATCH --job-name=s2-four-view-nw
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
AUDITOR=$DEPLOY_ROOT/jobs/interaction_first/audit_train_only_s2_exact_four_view_nw.py
AUTHORITY=${S2_NW_AUTHORITY_PATH:-$DEPLOY_ROOT/data/interaction_first/double_remote_split.json}
CANONICAL_MANIFEST=${S2_NW_CANONICAL_MANIFEST_PATH:-$DEPLOY_ROOT/data/interaction_first/canonical_manifest.csv}
ASSIGNMENT=${S2_NW_ASSIGNMENT_PATH:?Set S2_NW_ASSIGNMENT_PATH}
OUTPUT_DIR=${S2_NW_OUTPUT_DIR:?Set S2_NW_OUTPUT_DIR to a new create-only directory}
ASSIGNMENT_SHA256=${S2_NW_ASSIGNMENT_SHA256:?Set S2_NW_ASSIGNMENT_SHA256}
ASSIGNMENT_SCIENTIFIC_SHA256=${S2_NW_ASSIGNMENT_SCIENTIFIC_SHA256:?Set S2_NW_ASSIGNMENT_SCIENTIFIC_SHA256}
AUTHORITY_SHA256=${S2_NW_AUTHORITY_SHA256:?Set S2_NW_AUTHORITY_SHA256}
CANONICAL_MANIFEST_SHA256=${S2_NW_CANONICAL_MANIFEST_SHA256:?Set S2_NW_CANONICAL_MANIFEST_SHA256}
AUDITOR_SOURCE_SHA256=${S2_NW_AUDITOR_SOURCE_SHA256:?Set S2_NW_AUDITOR_SOURCE_SHA256}

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/usr/bin/time -v "$PYTHON_BIN" "$AUDITOR" \
  --assignment "$ASSIGNMENT" \
  --assignment-sha256 "$ASSIGNMENT_SHA256" \
  --assignment-scientific-sha256 "$ASSIGNMENT_SCIENTIFIC_SHA256" \
  --authority-split "$AUTHORITY" \
  --authority-sha256 "$AUTHORITY_SHA256" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --canonical-manifest-sha256 "$CANONICAL_MANIFEST_SHA256" \
  --expected-auditor-source-sha256 "$AUDITOR_SOURCE_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --chunk-size 512
