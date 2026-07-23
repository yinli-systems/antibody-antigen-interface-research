#!/bin/bash
#SBATCH --job-name=interaction-data
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
CANONICAL_MANIFEST=$BENCHMARK_ROOT/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
MMSEQS=$DEPLOY_ROOT/tools/mmseqs2-latest/bin/mmseqs
OUTPUT_ROOT=$DEPLOY_ROOT/data/interaction_first
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_interaction_first.py
MAPPING_DIR=$BENCHMARK_ROOT/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/mapping

mkdir -p "$OUTPUT_ROOT" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp

"$PYTHON_BIN" "$SCRIPT" prepare-contacts \
  --benchmark-root "$BENCHMARK_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --resume

"$PYTHON_BIN" "$SCRIPT" prepare-antibody-metadata \
  --benchmark-root "$BENCHMARK_ROOT" \
  --mapping-dir "$MAPPING_DIR" \
  --output "$OUTPUT_ROOT/antibody_cdrh3.csv"

"$PYTHON_BIN" "$SCRIPT" build-split \
  --benchmark-root "$BENCHMARK_ROOT" \
  --mmseqs "$MMSEQS" \
  --work-dir "$OUTPUT_ROOT/mmseqs" \
  --cdrh3-metadata "$OUTPUT_ROOT/antibody_cdrh3.csv" \
  --group-manifest "$CANONICAL_MANIFEST" \
  --group-id-column INSTANCE \
  --group-columns ab_ag_cluster \
  --antibody-identity 0.80 \
  --antigen-identity 0.70 \
  --coverage 0.80 \
  --threads 12 \
  --output "$OUTPUT_ROOT/double_remote_split.json"
