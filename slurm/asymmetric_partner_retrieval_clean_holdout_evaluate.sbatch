#!/bin/bash
#SBATCH --job-name=asym-retrieval-eval
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
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
SPLIT_MANIFEST=${RETRIEVAL_SPLIT_MANIFEST:?Set the reviewed NW-clean split}
SPLIT_SHA256=${RETRIEVAL_SPLIT_SHA256:?Set the reviewed NW-clean split SHA-256}
STRICT_NW_MODE=${RETRIEVAL_STRICT_NW_CONSTRAINT_MODE:?Set the explicit NW constraint mode}
CHECKPOINT=${RETRIEVAL_FINAL_CHECKPOINT:?Set the completed fixed-epoch final checkpoint}
CHECKPOINT_SHA256=${RETRIEVAL_FINAL_CHECKPOINT_SHA256:?Set the reviewed final checkpoint SHA-256}
VARIANT=${RETRIEVAL_VARIANT:?Set the fold-selected variant explicitly}
OUTPUT_JSON=${RETRIEVAL_EVALUATION_JSON:?Set a new one-shot evaluation JSON path}
SCORE_ARTIFACT=${RETRIEVAL_SCORE_ARTIFACT:?Set a new one-shot score NPZ path}
ONE_SHOT_LEDGER_REGISTRY_DIR=${RETRIEVAL_ONE_SHOT_LEDGER_REGISTRY_DIR:?Set the central immutable evaluation-ledger registry}
SEED=${RETRIEVAL_SEED:-42}

ANTIBERTY_WEIGHTS_SHA256=f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137
ESM_WEIGHTS_SHA256=a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0

if [[ "$VARIANT" != "antiberty_calm_mean" ]]; then
  echo "Final evaluation is locked to RETRIEVAL_VARIANT=antiberty_calm_mean" >&2
  exit 2
fi
if [[ "$SEED" != "42" ]]; then
  echo "Final evaluation is locked to RETRIEVAL_SEED=42" >&2
  exit 2
fi
ANTIBODY_DIR=$ANTIBERTY_DIR
ANTIBODY_FORMAT=antiberty
ANTIBODY_MODEL=AntiBERTy_md_smooth
ANTIBODY_WEIGHTS=$ANTIBERTY_WEIGHTS_SHA256
ANTIBODY_DIM=512

if [[ "$STRICT_NW_MODE" != "concat_antibody_and_antigen" && "$STRICT_NW_MODE" != "concat_antibody_antigen_plus_vh_vl" ]]; then
  echo "Unsupported RETRIEVAL_STRICT_NW_CONSTRAINT_MODE: $STRICT_NW_MODE" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" || "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" != "$CHECKPOINT_SHA256" ]]; then
  echo "Final checkpoint is missing or its SHA-256 does not match" >&2
  exit 4
fi
for output in "$OUTPUT_JSON" "$SCORE_ARTIFACT"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to reuse one-shot path: $output" >&2
    exit 3
  fi
  mkdir -p "$(dirname "$output")"
done
mkdir -p "$ONE_SHOT_LEDGER_REGISTRY_DIR"

mkdir -p "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$SCRIPT" evaluate \
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
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --output "$OUTPUT_JSON" \
  --score-artifact "$SCORE_ARTIFACT" \
  --one-shot-ledger-registry-dir "$ONE_SHOT_LEDGER_REGISTRY_DIR" \
  --antibody-artifact-format "$ANTIBODY_FORMAT" \
  --antibody-model-name "$ANTIBODY_MODEL" \
  --antibody-model-weights-sha256 "$ANTIBODY_WEIGHTS" \
  --antigen-model-name facebook/esm2_t33_650M_UR50D \
  --antigen-model-weights-sha256 "$ESM_WEIGHTS_SHA256" \
  --antibody-input-dim "$ANTIBODY_DIM" \
  --antigen-input-dim 1280 \
  --batch-size 256 \
  --bucket-width 128 \
  --recall-ks 1 5 10 20 \
  --seed "$SEED" \
  --device cuda
