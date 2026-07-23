#!/bin/bash
#SBATCH --job-name=nw-authority-boundary
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
DATA_ROOT=$DEPLOY_ROOT/data/interaction_first
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
MANIFEST=$BENCHMARK_ROOT/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
BOUNDARY_ROOT=$DATA_ROOT/authority_validation_nw_boundaries
AUTHORITY=$DATA_ROOT/double_remote_split.json

mkdir -p "$BOUNDARY_ROOT/independent_edges" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" "$SCRIPT_ROOT/verify_authority_validation_nw_boundaries.py" \
  --authority-split "$AUTHORITY" \
  --sequence-manifest "$MANIFEST" \
  --boundary "concat_antibody_and_antigen=$BOUNDARY_ROOT/concat_boundary_v2.json" \
  --boundary "concat_antibody_antigen_plus_vh_vl=$BOUNDARY_ROOT/chain_boundary_v2.json" \
  --workers 12 \
  --chunk-size 512 \
  --edge-output-dir "$BOUNDARY_ROOT/independent_edges" \
  --output "$BOUNDARY_ROOT/independent_verification_v2.json"
