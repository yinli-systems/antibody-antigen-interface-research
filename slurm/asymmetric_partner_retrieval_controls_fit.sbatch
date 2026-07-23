#!/bin/bash
#SBATCH --job-name=retrieval-controls-fit
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
CONTROL_SOURCE_ROOT=$DEPLOY_ROOT/jobs/interaction_first/controls_v1_6_schema_fix_bb574209/sources
SCRIPT=$CONTROL_SOURCE_ROOT/sabdab2_retrieval_controls.py
CONTROL_SOURCE_SHA256=bb5742097c9a0b0f46419b0c3bebd9ccd378c69dc4137483f55d28219ca9a285
PREPARE_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_prepared
PREPARED_INSTANCES_SHA256=a9131d52323bae40572b7de81c5e41353ccc39a73309b7d7043485350facb009
CANONICAL_SEQUENCE_MANIFEST=$DEPLOY_ROOT/data/interaction_first/canonical_manifest.csv
CANONICAL_SEQUENCE_MANIFEST_SHA256=548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63
ESM_DIR=$DEPLOY_ROOT/data/sppider_seq_sabdab2_embeddings
ANTIBERTY_DIR=$DEPLOY_ROOT/data/antiberty_sabdab2_development_embeddings
AUTHORITY_SPLIT=$DEPLOY_ROOT/data/interaction_first/double_remote_split.json
AUTHORITY_SHA256=e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b
BOUNDARY_VERIFICATION=${RETRIEVAL_CONTROL_BOUNDARY_VERIFICATION:?Set the independent v2 boundary verification manifest}
BOUNDARY_VERIFICATION_SHA256=3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8
CONTROL_PROTOCOL=${RETRIEVAL_CONTROL_PROTOCOL_V1:?Set the physical controls protocol v1 path}
CONTROL_PROTOCOL_SHA256=21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02
EXECUTION_PROTOCOL=${RETRIEVAL_PROTOCOL_V1_6:?Set the final physical protocol v1.6 path}
EXECUTION_PROTOCOL_SHA256=b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078
SPLIT_MANIFEST=${RETRIEVAL_CONTROL_SPLIT_MANIFEST:?Set the reviewed repaired development split}
SPLIT_SHA256=${RETRIEVAL_CONTROL_SPLIT_SHA256:?Set the reviewed repaired split SHA-256}
STRICT_NW_MODE=${RETRIEVAL_CONTROL_STRICT_NW_MODE:?Set the explicit NW constraint mode}
OUTPUT_DIR=${RETRIEVAL_CONTROL_FIT_OUTPUT_DIR:?Set a new train-only control-fit directory}

ANTIBERTY_WEIGHTS_SHA256=f1ae33eac8cc8784a7d4be5a600141d2fa7bc7d8d5b3f5324d64a6a63bd0f137
ESM_WEIGHTS_SHA256=a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0
BUNDLE=$OUTPUT_DIR/controls.bundle.npz
SUMMARY=$OUTPUT_DIR/fit.summary.json

if [[ ! -f "$SCRIPT" || "$(sha256sum "$SCRIPT" | awk '{print $1}')" != "$CONTROL_SOURCE_SHA256" ]]; then
  echo "The frozen control source is missing or changed" >&2
  exit 4
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse a control-fit directory: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ "$STRICT_NW_MODE" != "concat_antibody_and_antigen" && "$STRICT_NW_MODE" != "concat_antibody_antigen_plus_vh_vl" ]]; then
  echo "Unsupported RETRIEVAL_CONTROL_STRICT_NW_MODE: $STRICT_NW_MODE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$SCRIPT" fit \
  --prepare-dir "$PREPARE_DIR" \
  --prepared-instances-sha256 "$PREPARED_INSTANCES_SHA256" \
  --canonical-sequence-manifest "$CANONICAL_SEQUENCE_MANIFEST" \
  --canonical-sequence-manifest-sha256 "$CANONICAL_SEQUENCE_MANIFEST_SHA256" \
  --antibody-embedding-dir "$ANTIBERTY_DIR" \
  --shared-esm-embedding-dir "$ESM_DIR" \
  --antigen-embedding-dir "$ESM_DIR" \
  --split-manifest "$SPLIT_MANIFEST" \
  --split-manifest-sha256 "$SPLIT_SHA256" \
  --authority-split-manifest "$AUTHORITY_SPLIT" \
  --authority-split-sha256 "$AUTHORITY_SHA256" \
  --boundary-verification-manifest "$BOUNDARY_VERIFICATION" \
  --boundary-verification-sha256 "$BOUNDARY_VERIFICATION_SHA256" \
  --control-protocol "$CONTROL_PROTOCOL" \
  --control-protocol-sha256 "$CONTROL_PROTOCOL_SHA256" \
  --execution-protocol "$EXECUTION_PROTOCOL" \
  --execution-protocol-sha256 "$EXECUTION_PROTOCOL_SHA256" \
  --authorized-source-pool-splits train validation \
  --authority-sealed-splits test \
  --authority-coverage-mode exact \
  --strict-nw-constraint-mode "$STRICT_NW_MODE" \
  --isolation-policy double_exact \
  --antibody-artifact-format antiberty \
  --antibody-model-name AntiBERTy_md_smooth \
  --antibody-model-weights-sha256 "$ANTIBERTY_WEIGHTS_SHA256" \
  --antibody-input-dim 512 \
  --esm-model-name facebook/esm2_t33_650M_UR50D \
  --esm-model-weights-sha256 "$ESM_WEIGHTS_SHA256" \
  --esm-input-dim 1280 \
  --cache-gib 0 \
  --negative-replicas 5 \
  --negative-seed 42017 \
  --logistic-l2 0.01 \
  --logistic-max-iterations 50 \
  --logistic-tolerance 0.000000001 \
  --random-head-seed 20260723 \
  --random-head-hidden-dim 256 \
  --random-head-output-dim 128 \
  --output-bundle "$BUNDLE" \
  --output-summary "$SUMMARY"

sha256sum "$BUNDLE" "$SUMMARY"
