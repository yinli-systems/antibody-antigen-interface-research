#!/bin/bash
#SBATCH --job-name=asym-retrieval-stats
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_asymmetric_retrieval_statistics.py
EVALUATION_JSON=${RETRIEVAL_EVALUATION_JSON:?Set the completed clean-holdout evaluation JSON}
EVALUATION_SHA256=${RETRIEVAL_EVALUATION_SHA256:?Set the reviewed evaluation JSON SHA-256}
SCORE_ARTIFACT=${RETRIEVAL_SCORE_ARTIFACT:?Set the completed clean-holdout score NPZ}
SCORE_SHA256=${RETRIEVAL_SCORE_ARTIFACT_SHA256:?Set the reviewed score NPZ SHA-256}
OUTPUT_JSON=${RETRIEVAL_STATISTICS_OUTPUT:?Set a new statistics output path}
SEED=${RETRIEVAL_STATISTICS_SEED:-42}
PERMUTATIONS=${RETRIEVAL_PERMUTATIONS:-10000}
BOOTSTRAPS=${RETRIEVAL_BOOTSTRAPS:-10000}
CANDIDATE_SAMPLES=${RETRIEVAL_CANDIDATE_SAMPLES:-1000}

if [[ ! -f "$EVALUATION_JSON" || ! -f "$SCORE_ARTIFACT" ]]; then
  echo "The one-shot clean-holdout evaluation and score artifact must both exist" >&2
  exit 3
fi
if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite an existing statistics output: $OUTPUT_JSON" >&2
  exit 3
fi
if [[ "$(sha256sum "$EVALUATION_JSON" | awk '{print $1}')" != "$EVALUATION_SHA256" ]]; then
  echo "Evaluation JSON SHA-256 mismatch" >&2
  exit 4
fi
if [[ "$(sha256sum "$SCORE_ARTIFACT" | awk '{print $1}')" != "$SCORE_SHA256" ]]; then
  echo "Score artifact SHA-256 mismatch" >&2
  exit 4
fi

CLUSTER_ARGS=()
if [[ -n "${RETRIEVAL_ANCHOR_CLUSTERS_JSON:-}" || -n "${RETRIEVAL_ANCHOR_CLUSTERS_SHA256:-}" ]]; then
  : "${RETRIEVAL_ANCHOR_CLUSTERS_JSON:?Set both cluster path and SHA-256}"
  : "${RETRIEVAL_ANCHOR_CLUSTERS_SHA256:?Set both cluster path and SHA-256}"
  CLUSTER_ARGS=(
    --anchor-clusters-json "$RETRIEVAL_ANCHOR_CLUSTERS_JSON"
    --expected-anchor-clusters-sha256 "$RETRIEVAL_ANCHOR_CLUSTERS_SHA256"
  )
fi

mkdir -p "$(dirname "$OUTPUT_JSON")" "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$SCRIPT" \
  --evaluation-json "$EVALUATION_JSON" \
  --score-artifact "$SCORE_ARTIFACT" \
  --expected-evaluation-sha256 "$EVALUATION_SHA256" \
  --expected-score-artifact-sha256 "$SCORE_SHA256" \
  --output "$OUTPUT_JSON" \
  --permutations "$PERMUTATIONS" \
  --bootstrap-samples "$BOOTSTRAPS" \
  --seed "$SEED" \
  --candidate-sizes 10 25 50 100 \
  --candidate-samples "$CANDIDATE_SAMPLES" \
  "${CLUSTER_ARGS[@]}"
