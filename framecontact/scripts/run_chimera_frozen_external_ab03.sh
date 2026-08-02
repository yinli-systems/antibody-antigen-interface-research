#!/usr/bin/env bash
set -euo pipefail

readonly EVAL_BASE="${CHIMERA_EVAL_BASE:-/home/liyin/geometry-self-attention-frozen-eval-20260730}"
readonly SOURCE_DIR="${CHIMERA_SOURCE_DIR:-${EVAL_BASE}/source/geometry-chimera-eval-source-v4}"
readonly CACHE_DIR="${EVAL_BASE}/data/chimera-strict-adapter-cache"
readonly HF_CACHE="${EVAL_BASE}/models/huggingface/hub"
readonly OUTPUT_DIR="${EVAL_BASE}/outputs"
readonly LOG_DIR="${EVAL_BASE}/logs"
readonly CHECKPOINT_ROOT="/home/liyin/geometry-self-attention-20260729-v2"
readonly CACHE_MANIFEST_SHA256="ad45a5a901045784aa7fdf038bf5e3d0d1018a26eb4c107c9b48fdcf11bc62af"
readonly ESM_WEIGHT_SHA256="a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
readonly ESM_REVISION="08e4846e537177426273712802403f7ba8261b6c"
readonly ESM_SNAPSHOT="${HF_CACHE}/models--facebook--esm2_t33_650M_UR50D/snapshots/${ESM_REVISION}"
readonly EMBEDDING_DIR="${OUTPUT_DIR}/chimera-esm2-embeddings-v1"
readonly PREDICTION_DIR="${OUTPUT_DIR}/chimera-frozen-predictions-v1"
readonly RESULT_DIR="${OUTPUT_DIR}/chimera-strict-external-result-v1"
readonly ESM_IMAGE="baseline/sppider-seq:v2.1.0-241d08e-cuda12.4"
readonly EVAL_IMAGE="liyin/sabdab2-geometry-a100-replay:sklearn-1.6.1-872701e1"

test -d "${SOURCE_DIR}"
test -d "${CACHE_DIR}"
test -d "${CHECKPOINT_ROOT}"
test -f "${ESM_SNAPSHOT}/model.safetensors"
test "$(sha256sum "${ESM_SNAPSHOT}/model.safetensors" | cut -d' ' -f1)" = "${ESM_WEIGHT_SHA256}"
test "$(sha256sum "${CACHE_DIR}/ADAPTER_CACHE_MANIFEST.json" | cut -d' ' -f1)" = "${CACHE_MANIFEST_SHA256}"
test ! -e "${PREDICTION_DIR}"
test ! -e "${RESULT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

docker run --rm --gpus 'device=0' --entrypoint sh \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -v "${SOURCE_DIR}:/work:ro" \
  -v "${CACHE_DIR}:/cache:ro" \
  -v "${HF_CACHE}:/hf:ro" \
  -v "${OUTPUT_DIR}:/outputs" \
  -w /work "${ESM_IMAGE}" -lc \
  "python scripts/materialize_chimera_esm_embeddings.py \
    --cache-root /cache \
    --expected-cache-manifest-sha256 ${CACHE_MANIFEST_SHA256} \
    --embedding-root /outputs/chimera-esm2-embeddings-v1 \
    --hf-cache /hf \
    --device cuda \
    --resume" \
  2>&1 | tee "${LOG_DIR}/chimera-esm2-materialization.log"

docker run --rm --gpus 'device=0' --entrypoint sh \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e PYTHONPATH=/work/src \
  -v "${SOURCE_DIR}:/work:ro" \
  -v "${CACHE_DIR}:/cache:ro" \
  -v "${OUTPUT_DIR}:/outputs" \
  -v "${CHECKPOINT_ROOT}:${CHECKPOINT_ROOT}:ro" \
  -w /work "${EVAL_IMAGE}" -lc \
  "python scripts/predict_chimera_frozen_external.py \
    --protocol /work/evaluation/chimera_zero_shot_protocol_v1.json \
    --freeze-receipt /work/evaluation/AB03_FROZEN_PROTOCOL_RECEIPT.json \
    --cache-root /cache \
    --expected-cache-manifest-sha256 ${CACHE_MANIFEST_SHA256} \
    --embedding-root /outputs/chimera-esm2-embeddings-v1 \
    --output-root /outputs/chimera-frozen-predictions-v1 \
    --device cuda \
    --precision bf16" \
  2>&1 | tee "${LOG_DIR}/chimera-frozen-prediction.log"

PREDICTION_MANIFEST_SHA256="$(
  sha256sum "${PREDICTION_DIR}/PREDICTION_MANIFEST.json" | cut -d' ' -f1
)"
readonly PREDICTION_MANIFEST_SHA256
test -n "${PREDICTION_MANIFEST_SHA256}"
docker run --rm --entrypoint sh \
  -e PYTHONPATH=/work/src \
  -v "${SOURCE_DIR}:/work:ro" \
  -v "${CACHE_DIR}:/cache:ro" \
  -v "${OUTPUT_DIR}:/outputs" \
  -w /work "${EVAL_IMAGE}" -lc \
  "python scripts/score_chimera_frozen_external.py \
    --protocol /work/evaluation/chimera_zero_shot_protocol_v1.json \
    --freeze-receipt /work/evaluation/AB03_FROZEN_PROTOCOL_RECEIPT.json \
    --adapter-audit /work/evaluation/CHIMERA_PRIMARY_ADAPTER_AUDIT.json \
    --cache-root /cache \
    --expected-cache-manifest-sha256 ${CACHE_MANIFEST_SHA256} \
    --prediction-root /outputs/chimera-frozen-predictions-v1 \
    --expected-prediction-manifest-sha256 ${PREDICTION_MANIFEST_SHA256} \
    --output-root /outputs/chimera-strict-external-result-v1" \
  2>&1 | tee "${LOG_DIR}/chimera-strict-external-scoring.log"

sha256sum \
  "${EMBEDDING_DIR}/CHIMERA_ESM_EMBEDDING_MANIFEST.json" \
  "${PREDICTION_DIR}/PREDICTION_MANIFEST.json" \
  "${RESULT_DIR}/CHIMERA_STRICT_EXTERNAL_RESULT.json" \
  "${RESULT_DIR}/CHIMERA_STRICT_EXTERNAL_RESULT.md" \
  | tee "${LOG_DIR}/chimera-frozen-final-sha256.txt"
