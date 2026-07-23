#!/bin/bash
#SBATCH --job-name=retrieval-control-stats
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
SCRIPT=$DEPLOY_ROOT/jobs/interaction_first/sabdab2_asymmetric_retrieval_statistics.py
CONTROL_NAME=${RETRIEVAL_CONTROL_NAME:?Set one locked control name}
EVALUATION_JSON=${RETRIEVAL_CONTROL_EVALUATION_JSON:?Set the authenticated evaluation JSON}
EVALUATION_SHA256=${RETRIEVAL_CONTROL_EVALUATION_SHA256:?Set its exact SHA-256}
SCORE_ARTIFACT=${RETRIEVAL_CONTROL_SCORE_ARTIFACT:?Set the authenticated score NPZ}
SCORE_SHA256=${RETRIEVAL_CONTROL_SCORE_SHA256:?Set its exact SHA-256}
OUTPUT_JSON=${RETRIEVAL_CONTROL_STATISTICS_OUTPUT:?Set a new statistics output path}

case "$CONTROL_NAME" in
  raw_shared_esm_mean_cosine|random_untrained_calm_head|antibody_only_unary|antigen_only_unary|length_composition_logistic) ;;
  *)
    echo "Unsupported RETRIEVAL_CONTROL_NAME: $CONTROL_NAME" >&2
    exit 2
    ;;
esac
if [[ -e "$OUTPUT_JSON" ]]; then
  echo "Refusing to overwrite control statistics: $OUTPUT_JSON" >&2
  exit 3
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
  --permutations 10000 \
  --bootstrap-samples 10000 \
  --alpha 0.05 \
  --seed 42 \
  --candidate-sizes 10 25 50 100 \
  --candidate-samples 1000

sha256sum "$OUTPUT_JSON"
