#!/bin/bash
#SBATCH --job-name=identity80-dev
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
MMSEQS=$DEPLOY_ROOT/tools/mmseqs2-latest/bin/mmseqs
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/build_identity_isolated_development_split.py
BENCHMARK_ROOT=$DEPLOY_ROOT/data/sabdab2-baseline-outsourcing
MANIFEST=$BENCHMARK_ROOT/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
SOURCE_SPLIT=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
OUTPUT_ROOT=$DEPLOY_ROOT/data/interaction_first/identity80_development
SEARCH_ROOT=$OUTPUT_ROOT/exhaustive_mmseqs

mkdir -p "$OUTPUT_ROOT" "$SEARCH_ROOT" "$DEPLOY_ROOT/logs"
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" "$SCRIPT" export-fasta \
  --manifest "$MANIFEST" \
  --source-split "$SOURCE_SPLIT" \
  --pool-splits train validation \
  --sealed-splits test \
  --output-dir "$SEARCH_ROOT"

for ROLE in antibody antigen; do
  FASTA=$SEARCH_ROOT/development_${ROLE}.fasta
  RESULT=$SEARCH_ROOT/${ROLE}_identity80_exhaustive.tsv
  TEMP_DIR=$SEARCH_ROOT/${ROLE}_tmp
  "$MMSEQS" easy-search \
    "$FASTA" "$FASTA" "$RESULT" "$TEMP_DIR" \
    --format-output query,target,fident,qcov,tcov \
    --min-seq-id 0.80 \
    -c 0.80 \
    --cov-mode 0 \
    --alignment-mode 3 \
    --prefilter-mode 2 \
    --mask 0 \
    --comp-bias-corr 0 \
    --max-seqs 10000 \
    -e inf \
    --threads 12
done

"$PYTHON_BIN" "$SCRIPT" build \
  --manifest "$MANIFEST" \
  --source-split "$SOURCE_SPLIT" \
  --pool-splits train validation \
  --sealed-splits test \
  --antibody-hits "$SEARCH_ROOT/antibody_identity80_exhaustive.tsv" \
  --antigen-hits "$SEARCH_ROOT/antigen_identity80_exhaustive.tsv" \
  --identity-threshold 0.80 \
  --coverage-threshold 0.80 \
  --antibody-artifact-min-identity 0.80 \
  --antigen-artifact-min-identity 0.80 \
  --artifact-min-coverage 0.80 \
  --validation-fraction 0.10 \
  --minimum-validation-count 200 \
  --minimum-train-count 2500 \
  --seed 42 \
  --output "$OUTPUT_ROOT/split.json"

"$PYTHON_BIN" "$SCRIPT" audit \
  --manifest "$MANIFEST" \
  --split "$OUTPUT_ROOT/split.json" \
  --sealed-splits \
  --antibody-hits "$SEARCH_ROOT/antibody_identity80_exhaustive.tsv" \
  --antigen-hits "$SEARCH_ROOT/antigen_identity80_exhaustive.tsv" \
  --identity-threshold 0.80 \
  --coverage-threshold 0.80 \
  --antibody-artifact-min-identity 0.80 \
  --antigen-artifact-min-identity 0.80 \
  --artifact-min-coverage 0.80 \
  --output "$OUTPUT_ROOT/audit.json"
