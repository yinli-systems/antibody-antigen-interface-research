#!/bin/bash
#SBATCH --job-name=nw-strict-dev
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
SEARCH_ROOT=$IDENTITY_ROOT/exhaustive_mmseqs
AUDIT_REPORT=$IDENTITY_ROOT/needleman_wunsch_audit.json
STRICT_SPLIT=$IDENTITY_ROOT/nw_strict_split.json

mkdir -p "$IDENTITY_ROOT" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PYTHON_BIN" "$SCRIPT_ROOT/test_needleman_wunsch_development_audit.py" -v
"$PYTHON_BIN" "$SCRIPT_ROOT/test_reconcile_mmseqs_needleman_wunsch.py" -v
"$PYTHON_BIN" "$SCRIPT_ROOT/test_build_iterative_nw_strict_development_split.py" -v

"$PYTHON_BIN" "$SCRIPT_ROOT/needleman_wunsch_development_audit.py" \
  --manifest "$MANIFEST" \
  --development-source-split "$SOURCE_SPLIT" \
  --audit-split "active=$SOURCE_SPLIT" \
  --audit-split "identity80=$IDENTITY_ROOT/split.json" \
  --pool-splits train validation \
  --sealed-splits test \
  --workers 12 \
  --edge-output-dir "$IDENTITY_ROOT/needleman_wunsch_edges" \
  --output "$AUDIT_REPORT"

"$PYTHON_BIN" "$SCRIPT_ROOT/reconcile_mmseqs_needleman_wunsch.py" \
  --nw-report "$AUDIT_REPORT" \
  --antibody-mmseqs "$SEARCH_ROOT/antibody_identity80_exhaustive.tsv" \
  --antigen-mmseqs "$SEARCH_ROOT/antigen_identity80_exhaustive.tsv" \
  --antibody-fasta "$SEARCH_ROOT/development_antibody.fasta" \
  --antigen-fasta "$SEARCH_ROOT/development_antigen.fasta" \
  --output "$IDENTITY_ROOT/mmseqs_nw_reconciliation.json"

"$PYTHON_BIN" "$SCRIPT_ROOT/build_iterative_nw_strict_development_split.py" \
  --manifest "$MANIFEST" \
  --development-source-split "$SOURCE_SPLIT" \
  --base-split "$IDENTITY_ROOT/split.json" \
  --antibody-hits "$SEARCH_ROOT/antibody_identity80_exhaustive.tsv" \
  --antigen-hits "$SEARCH_ROOT/antigen_identity80_exhaustive.tsv" \
  --pool-splits train validation \
  --sealed-splits test \
  --validation-fraction 0.10 \
  --minimum-validation-count 1 \
  --minimum-train-count 2500 \
  --seed 42 \
  --workers 12 \
  --maximum-iterations 10 \
  --edge-output-dir "$IDENTITY_ROOT/nw_strict_build_edges" \
  --output "$STRICT_SPLIT"

"$PYTHON_BIN" "$SCRIPT_ROOT/needleman_wunsch_development_audit.py" \
  --manifest "$MANIFEST" \
  --development-source-split "$SOURCE_SPLIT" \
  --audit-split "nw_strict=$STRICT_SPLIT" \
  --pool-splits train validation \
  --sealed-splits test \
  --workers 12 \
  --edge-output-dir "$IDENTITY_ROOT/nw_strict_verification_edges" \
  --output "$IDENTITY_ROOT/nw_strict_split_verification.json"
