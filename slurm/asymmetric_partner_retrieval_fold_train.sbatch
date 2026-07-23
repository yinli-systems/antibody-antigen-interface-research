#!/bin/bash
#SBATCH --job-name=asym-retrieval-fold
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%A_%a.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%A_%a.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_asymmetric_partner_retrieval.py
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
ESM_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
ANTIBERTY_DIR=$DEPLOY_ROOT/data/antiberty_sabdab2_development_embeddings
AUTHORITY_SPLIT=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
AUTHORITY_SHA256=e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b
FOLD_ROOT=${RETRIEVAL_FOLD_ROOT:-$DEPLOY_ROOT/data/interaction_first/antigen_cluster_development_folds}
FOLD_INDEX=${RETRIEVAL_FOLD_INDEX:-${SLURM_ARRAY_TASK_ID}}
FOLD_INDEX_MANIFEST=$FOLD_ROOT/index.json
FOLD_INDEX_SHA256=${RETRIEVAL_FOLD_INDEX_SHA256:?Set RETRIEVAL_FOLD_INDEX_SHA256 to the reviewed index hash}
VARIANT=${RETRIEVAL_VARIANT:-antiberty_calm_mean}
SEED=${RETRIEVAL_SEED:-42}
BATCH_SIZE=${RETRIEVAL_BATCH_SIZE:-1024}

ANTIBERTY_WEIGHTS_SHA256=f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137
ESM_WEIGHTS_SHA256=a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0

case "$VARIANT" in
  antiberty_calm_mean)
    ANTIBODY_DIR=$ANTIBERTY_DIR
    ANTIBODY_FORMAT=antiberty
    ANTIBODY_MODEL=AntiBERTy_md_smooth
    ANTIBODY_WEIGHTS=$ANTIBERTY_WEIGHTS_SHA256
    ANTIBODY_DIM=512
    CHAIN_FLAG=--no-chain-balanced-antibody-pooling
    ;;
  antiberty_chain_balanced)
    ANTIBODY_DIR=$ANTIBERTY_DIR
    ANTIBODY_FORMAT=antiberty
    ANTIBODY_MODEL=AntiBERTy_md_smooth
    ANTIBODY_WEIGHTS=$ANTIBERTY_WEIGHTS_SHA256
    ANTIBODY_DIM=512
    CHAIN_FLAG=--chain-balanced-antibody-pooling
    ;;
  esm2_calm_mean)
    ANTIBODY_DIR=$ESM_DIR
    ANTIBODY_FORMAT=esm_chunks
    ANTIBODY_MODEL=facebook/esm2_t33_650M_UR50D
    ANTIBODY_WEIGHTS=$ESM_WEIGHTS_SHA256
    ANTIBODY_DIM=1280
    CHAIN_FLAG=--no-chain-balanced-antibody-pooling
    ;;
  *)
    echo "Unsupported RETRIEVAL_VARIANT: $VARIANT" >&2
    exit 2
    ;;
esac

RUN_NAME=${VARIANT}_antigen_fold${FOLD_INDEX}_seed${SEED}_b${BATCH_SIZE}
OUTPUT_DIR=$DEPLOY_ROOT/results/sabdab2/asymmetric_partner_retrieval/$RUN_NAME/checkpoints

if [[ ! -f "$FOLD_INDEX_MANIFEST" ]]; then
  echo "Fold index is missing: $FOLD_INDEX_MANIFEST" >&2
  exit 3
fi
if [[ ! -f "$AUTHORITY_SPLIT" ]]; then
  echo "Authoritative source split is missing: $AUTHORITY_SPLIT" >&2
  exit 3
fi
if [[ "$BATCH_SIZE" != "1024" && "$BATCH_SIZE" != "512" ]]; then
  echo "RETRIEVAL_BATCH_SIZE must be an explicitly recorded 1024 or 512" >&2
  exit 2
fi

read -r FOLD_MANIFEST FOLD_SHA256 < <(
  "$PYTHON_BIN" -c '
import hashlib, json, pathlib, sys
index_path = pathlib.Path(sys.argv[1]).resolve()
expected_index_hash = sys.argv[2]
observed_index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
if observed_index_hash != expected_index_hash:
    raise SystemExit("Fold index SHA-256 mismatch")
payload = json.loads(index_path.read_text(encoding="utf-8"))
fold_index = int(sys.argv[3])
matches = [entry for entry in payload.get("folds", []) if int(entry.get("fold_index", -1)) == fold_index]
if len(matches) != 1:
    raise SystemExit("Fold index must contain exactly one requested fold")
entry = matches[0]
path = (index_path.parent / entry["path"]).resolve()
digest = str(entry["sha256"])
if len(digest) != 64:
    raise SystemExit("Fold manifest SHA-256 is invalid")
print(path, digest)
' "$FOLD_INDEX_MANIFEST" "$FOLD_INDEX_SHA256" "$FOLD_INDEX"
)
if [[ ! -f "$FOLD_MANIFEST" ]]; then
  echo "Fold manifest is missing: $FOLD_MANIFEST" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

/usr/bin/time -v "$PYTHON_BIN" "$SCRIPT" train \
  --prepare-dir "$PREPARE_DIR" \
  --antibody-embedding-dir "$ANTIBODY_DIR" \
  --antigen-embedding-dir "$ESM_DIR" \
  --split-manifest "$FOLD_MANIFEST" \
  --split-manifest-sha256 "$FOLD_SHA256" \
  --authority-split-manifest "$AUTHORITY_SPLIT" \
  --authority-split-sha256 "$AUTHORITY_SHA256" \
  --authorized-source-pool-splits train \
  --authority-sealed-splits validation test \
  --authority-coverage-mode exact \
  --isolation-policy antigen_exact \
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
  --validation-batch-size 256 \
  --learning-rate 0.001 \
  --minimum-learning-rate 0.000001 \
  --weight-decay 0.20 \
  --schedule cosine_warm_restarts \
  --restart-period-epochs 10 \
  --restart-multiplier 2 \
  --gradient-clip 1.0 \
  --max-epochs 100 \
  --patience 15 \
  --min-delta 0.0001 \
  --seed "$SEED" \
  --device cuda \
  --resume
