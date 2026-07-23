#!/bin/bash
#SBATCH --job-name=nw-dev-audit
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT_ROOT=$DEPLOY_ROOT/jobs/interaction_first
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
MANIFEST=$BENCHMARK_ROOT/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
SOURCE_SPLIT=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
IDENTITY_ROOT=$DEPLOY_ROOT/data/interaction_first/identity80_development
OUTPUT=$IDENTITY_ROOT/needleman_wunsch_audit.json

mkdir -p "$IDENTITY_ROOT" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" "$SCRIPT_ROOT/test_needleman_wunsch_development_audit.py" -v

"$PYTHON_BIN" "$SCRIPT_ROOT/needleman_wunsch_development_audit.py" \
  --manifest "$MANIFEST" \
  --development-source-split "$SOURCE_SPLIT" \
  --audit-split "active=$SOURCE_SPLIT" \
  --audit-split "identity80=$IDENTITY_ROOT/split.json" \
  --pool-splits train validation \
  --sealed-splits test \
  --identity-threshold 0.80 \
  --coverage-threshold 0.80 \
  --calm-scaled-threshold 0.80 \
  --gap-open 10 \
  --gap-extend 1 \
  --workers 12 \
  --chunk-size 512 \
  --top-pair-count 20 \
  --edge-output-dir "$IDENTITY_ROOT/needleman_wunsch_edges" \
  --output "$OUTPUT"
