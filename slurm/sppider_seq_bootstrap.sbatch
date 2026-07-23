#!/bin/bash
#SBATCH --job-name=sppider-bootstrap
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON=$DEPLOY_ROOT/envs/epiformer/bin/python
ROLE=${SPPIDER_ROLE:?Set SPPIDER_ROLE to paratope or epitope}
RUN_NAME=${SPPIDER_RUN_NAME:-trained_seed42}
RESULT_ROOT=$DEPLOY_ROOT/results/sabdab2/sppider_seq/$RUN_NAME/$ROLE/final
CANONICAL_MANIFEST=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
OUTPUT=${SPPIDER_BOOTSTRAP_OUTPUT:-$RESULT_ROOT/cluster_bootstrap.json}
THRESHOLD=${SPPIDER_THRESHOLD:-0.50}

mkdir -p "$(dirname "$OUTPUT")" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export PATH=$DEPLOY_ROOT/envs/epiformer/bin:$PATH

/usr/bin/time -v "$PYTHON" "$DEPLOY_ROOT/jobs/sppider_seq/sppider_seq_research.py" cluster-bootstrap \
  --residue-predictions "$RESULT_ROOT/residue_predictions.parquet" \
  --canonical-manifest "$CANONICAL_MANIFEST" \
  --role "$ROLE" \
  --output "$OUTPUT" \
  --replicates 1000 \
  --seed 20260722 \
  --threshold "$THRESHOLD" \
  --calibration-bins 15
