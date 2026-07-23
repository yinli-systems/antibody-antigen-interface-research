#!/bin/bash
#SBATCH --job-name=retrieval-control-compare
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_retrieval_control_comparator.py

PRIMARY_EVALUATION=${RETRIEVAL_PRIMARY_EVALUATION_JSON:?Set the final-refit evaluation JSON}
PRIMARY_EVALUATION_SHA256=${RETRIEVAL_PRIMARY_EVALUATION_SHA256:?Set its exact SHA-256}
PRIMARY_SCORES=${RETRIEVAL_PRIMARY_SCORE_ARTIFACT:?Set the final-refit score artifact}
PRIMARY_SCORES_SHA256=${RETRIEVAL_PRIMARY_SCORE_ARTIFACT_SHA256:?Set its exact SHA-256}
CONTROL_DIR=${RETRIEVAL_CONTROL_SUITE_DIR:?Set the completed control-suite output directory}

RAW_EVALUATION_SHA256=${RETRIEVAL_RAW_ESM_EVALUATION_SHA256:?Set the raw-ESM evaluation SHA-256}
RAW_SCORES_SHA256=${RETRIEVAL_RAW_ESM_SCORES_SHA256:?Set the raw-ESM score SHA-256}
RANDOM_EVALUATION_SHA256=${RETRIEVAL_RANDOM_HEAD_EVALUATION_SHA256:?Set the random-head evaluation SHA-256}
RANDOM_SCORES_SHA256=${RETRIEVAL_RANDOM_HEAD_SCORES_SHA256:?Set the random-head score SHA-256}
ANTIBODY_EVALUATION_SHA256=${RETRIEVAL_ANTIBODY_UNARY_EVALUATION_SHA256:?Set the antibody-unary evaluation SHA-256}
ANTIBODY_SCORES_SHA256=${RETRIEVAL_ANTIBODY_UNARY_SCORES_SHA256:?Set the antibody-unary score SHA-256}
ANTIGEN_EVALUATION_SHA256=${RETRIEVAL_ANTIGEN_UNARY_EVALUATION_SHA256:?Set the antigen-unary evaluation SHA-256}
ANTIGEN_SCORES_SHA256=${RETRIEVAL_ANTIGEN_UNARY_SCORES_SHA256:?Set the antigen-unary score SHA-256}
LENGTH_EVALUATION_SHA256=${RETRIEVAL_LENGTH_COMPOSITION_EVALUATION_SHA256:?Set the length/composition evaluation SHA-256}
LENGTH_SCORES_SHA256=${RETRIEVAL_LENGTH_COMPOSITION_SCORES_SHA256:?Set the length/composition score SHA-256}

EXECUTION_PROTOCOL=${RETRIEVAL_PROTOCOL_V1_5:?Set the physical protocol v1.5 path}
CONTROL_PROTOCOL=${RETRIEVAL_CONTROLS_PROTOCOL_V1:?Set the physical controls protocol v1 path}
OUTPUT_JSON=${RETRIEVAL_CONTROL_COMPARISON_OUTPUT:?Set a new comparison output path}

EXECUTION_PROTOCOL_SHA256=43638fdd17b9ce3e4cac8b83e42c16261ecbf79c2e628d0b494e5dcbe02f9fc9
CONTROL_PROTOCOL_SHA256=21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02

if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite control comparison: $OUTPUT_JSON" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUTPUT_JSON")" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$SCRIPT" \
  --primary-evaluation-json "$PRIMARY_EVALUATION" \
  --primary-evaluation-sha256 "$PRIMARY_EVALUATION_SHA256" \
  --primary-score-artifact "$PRIMARY_SCORES" \
  --primary-score-artifact-sha256 "$PRIMARY_SCORES_SHA256" \
  --control raw_shared_esm_mean_cosine \
    "$CONTROL_DIR/raw_shared_esm_mean_cosine.evaluation.json" \
    "$RAW_EVALUATION_SHA256" \
    "$CONTROL_DIR/raw_shared_esm_mean_cosine.scores.npz" \
    "$RAW_SCORES_SHA256" \
  --control random_untrained_calm_head \
    "$CONTROL_DIR/random_untrained_calm_head.evaluation.json" \
    "$RANDOM_EVALUATION_SHA256" \
    "$CONTROL_DIR/random_untrained_calm_head.scores.npz" \
    "$RANDOM_SCORES_SHA256" \
  --control antibody_only_unary \
    "$CONTROL_DIR/antibody_only_unary.evaluation.json" \
    "$ANTIBODY_EVALUATION_SHA256" \
    "$CONTROL_DIR/antibody_only_unary.scores.npz" \
    "$ANTIBODY_SCORES_SHA256" \
  --control antigen_only_unary \
    "$CONTROL_DIR/antigen_only_unary.evaluation.json" \
    "$ANTIGEN_EVALUATION_SHA256" \
    "$CONTROL_DIR/antigen_only_unary.scores.npz" \
    "$ANTIGEN_SCORES_SHA256" \
  --control length_composition_logistic \
    "$CONTROL_DIR/length_composition_logistic.evaluation.json" \
    "$LENGTH_EVALUATION_SHA256" \
    "$CONTROL_DIR/length_composition_logistic.scores.npz" \
    "$LENGTH_SCORES_SHA256" \
  --execution-protocol "$EXECUTION_PROTOCOL" \
  --execution-protocol-sha256 "$EXECUTION_PROTOCOL_SHA256" \
  --control-protocol "$CONTROL_PROTOCOL" \
  --control-protocol-sha256 "$CONTROL_PROTOCOL_SHA256" \
  --output "$OUTPUT_JSON"

sha256sum "$OUTPUT_JSON"
