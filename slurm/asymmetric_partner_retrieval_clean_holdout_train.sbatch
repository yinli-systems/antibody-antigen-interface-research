#!/bin/bash
#SBATCH --job-name=asym-retrieval-clean
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_asymmetric_partner_retrieval.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
PREPARED_INSTANCES_SHA256=a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009
CANONICAL_SEQUENCE_MANIFEST=/cluster/project/model_deployment/data/sabdab2-baseline-outsourcing/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv
CANONICAL_SEQUENCE_MANIFEST_SHA256=548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63
ESM_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
ANTIBERTY_DIR=$DEPLOY_ROOT/data/antiberty_sabdab2_development_embeddings
AUTHORITY_SPLIT=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
AUTHORITY_SHA256=e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b
SUBMISSION_MANIFEST=${RETRIEVAL_SUBMISSION_MANIFEST:?Set the reviewed final-refit submission manifest}
SUBMISSION_MANIFEST_SHA256=${RETRIEVAL_SUBMISSION_MANIFEST_SHA256:?Set its reviewed SHA-256}
SPLIT_MANIFEST=${RETRIEVAL_SPLIT_MANIFEST:?Set RETRIEVAL_SPLIT_MANIFEST to a completed NW-clean split}
SPLIT_SHA256=${RETRIEVAL_SPLIT_SHA256:?Set RETRIEVAL_SPLIT_SHA256 to the reviewed split hash}
STRICT_NW_MODE=${RETRIEVAL_STRICT_NW_CONSTRAINT_MODE:?Set RETRIEVAL_STRICT_NW_CONSTRAINT_MODE explicitly}
SPLIT_TAG=${RETRIEVAL_SPLIT_TAG:-nw_clean}
VARIANT=${RETRIEVAL_VARIANT:-antiberty_calm_mean}
SEED=${RETRIEVAL_SEED:-42}
BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-1024}
FIXED_EPOCHS=${RETRIEVAL_FIXED_EPOCHS:?Set RETRIEVAL_FIXED_EPOCHS from the locked fold-selection rule}

ANTIBERTY_WEIGHTS_SHA256=f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137
ESM_WEIGHTS_SHA256=a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0

if [[ "$VARIANT" != "antiberty_calm_mean" ]]; then
  echo "Final refit is locked to RETRIEVAL_VARIANT=antiberty_calm_mean" >&2
  exit 2
fi
ANTIBODY_DIR=$ANTIBERTY_DIR
ANTIBODY_FORMAT=antiberty
ANTIBODY_MODEL=AntiBERTy_md_smooth
ANTIBODY_WEIGHTS=$ANTIBERTY_WEIGHTS_SHA256
ANTIBODY_DIM=512
CHAIN_FLAG=--no-chain-balanced-antibody-pooling

if [[ ! -f "$SPLIT_MANIFEST" ]]; then
  echo "NW-clean split is missing: $SPLIT_MANIFEST" >&2
  exit 3
fi
if [[ ! -f "$SUBMISSION_MANIFEST" || "$(sha256sum "$SUBMISSION_MANIFEST" | awk '{print $1}')" != "$SUBMISSION_MANIFEST_SHA256" ]]; then
  echo "Final-refit submission manifest is missing or its SHA-256 does not match" >&2
  exit 4
fi
if [[ "$BATCH_SIZE" != "1024" ]]; then
  echo "Final refit is locked to RETRIEVAL_BATCH_SIZE=1024" >&2
  exit 2
fi
if [[ "$FIXED_EPOCHS" != "58" ]]; then
  echo "Final refit is locked to RETRIEVAL_FIXED_EPOCHS=58" >&2
  exit 2
fi
if [[ "$SEED" != "42" ]]; then
  echo "Final refit is locked to RETRIEVAL_SEED=42" >&2
  exit 2
fi
if [[ "$STRICT_NW_MODE" != "concat_antibody_and_antigen" && "$STRICT_NW_MODE" != "concat_antibody_antigen_plus_vh_vl" ]]; then
  echo "Unsupported RETRIEVAL_STRICT_NW_CONSTRAINT_MODE: $STRICT_NW_MODE" >&2
  exit 2
fi

RUN_NAME=${VARIANT}_${SPLIT_TAG}_seed${SEED}_b${BATCH_SIZE}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/asymmetric_partner_retrieval/$RUN_NAME/checkpoints
mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" final-refit \
  --submission-manifest "$SUBMISSION_MANIFEST" \
  --submission-manifest-sha256 "$SUBMISSION_MANIFEST_SHA256" \
  --prepare-dir "$PREPARE_DIR" \
  --prepared-instances-sha256 "$PREPARED_INSTANCES_SHA256" \
  --canonical-sequence-manifest "$CANONICAL_SEQUENCE_MANIFEST" \
  --canonical-sequence-manifest-sha256 "$CANONICAL_SEQUENCE_MANIFEST_SHA256" \
  --antibody-embedding-dir "$ANTIBODY_DIR" \
  --antigen-embedding-dir "$ESM_DIR" \
  --split-manifest "$SPLIT_MANIFEST" \
  --split-manifest-sha256 "$SPLIT_SHA256" \
  --authority-split-manifest "$AUTHORITY_SPLIT" \
  --authority-split-sha256 "$AUTHORITY_SHA256" \
  --authorized-source-pool-splits train validation \
  --authority-sealed-splits test \
  --authority-coverage-mode exact \
  --strict-nw-constraint-mode "$STRICT_NW_MODE" \
  --isolation-policy double_exact \
  --output-dir "$OUTPUT_DIR" \
  --antibody-artifact-format "$ANTIBODY_FORMAT" \
  --antibody-model-name "$ANTIBODY_MODEL" \
  --antibody-model-weights-sha256 "$ANTIBODY_WEIGHTS" \
  --antigen-model-name facebook/esm2_t33_650M_UR50D \
  --antigen-model-weights-sha256 "$ESM_WEIGHTS_SHA256" \
  --antibody-input-dim "$ANTIBODY_DIM" \
  --antigen-input-dim 1280 \
  --embedding-dim 128 \
  --antibody-hidden-dim 256 \
  --antigen-hidden-dim 256 \
  --dropout 0.10 \
  "$CHAIN_FLAG" \
  --antibody-pooling mean \
  --antigen-pooling mean \
  --initial-temperature 0.20 \
  --minimum-temperature 0.01 \
  --maximum-temperature 1.00 \
  --batch-size "$BATCH_SIZE" \
  --learning-rate 0.001 \
  --minimum-learning-rate 0.000001 \
  --weight-decay 0.20 \
  --schedule cosine_warm_restarts \
  --warmup-fraction 0.10 \
  --restart-period-epochs 10 \
  --restart-multiplier 2 \
  --gradient-clip 1.0 \
  --fixed-epochs "$FIXED_EPOCHS" \
  --seed "$SEED" \
  --device cuda \
  --resume
