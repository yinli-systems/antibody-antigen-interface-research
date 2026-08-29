#!/usr/bin/env bash
set -euo pipefail

readonly EVAL_BASE="${CHIMERA_EVAL_BASE:-/home/liyin/geometry-self-attention-frozen-eval-20260730}"
readonly SOURCE_ARCHIVE_DIR="${EVAL_BASE}/source/geometry-chimera-eval-source-v8"
readonly SOURCE_DIR="${CHIMERA_SOURCE_DIR:-${SOURCE_ARCHIVE_DIR}/intra-chain-geometry-attention}"
readonly CACHE_DIR="${EVAL_BASE}/data/chimera-strict-adapter-cache"
readonly PREDICTION_DIR="${EVAL_BASE}/outputs/chimera-frozen-predictions-v1"
readonly OUTPUT_DIR="${EVAL_BASE}/outputs"
readonly LOG_DIR="${EVAL_BASE}/logs"
readonly CACHE_MANIFEST_SHA256="ad45a5a901045784aa7fdf038bf5e3d0d1018a26eb4c107c9b48fdcf11bc62af"
readonly PREDICTION_MANIFEST_SHA256="d06606ab7210cda31891846d2cac621a9dd0ac1a6050c3832ad5587fad5661cb"
readonly EVAL_IMAGE="liyin/sabdab2-geometry-a100-replay:sklearn-1.6.1-872701e1"
readonly CASE_STUDY_VERSION="${CHIMERA_CASE_STUDY_VERSION:-v2}"

test -d "${SOURCE_DIR}"
test -d "${CACHE_DIR}"
test -d "${PREDICTION_DIR}"
test "$(sha256sum "${CACHE_DIR}/ADAPTER_CACHE_MANIFEST.json" | cut -d' ' -f1)" = "${CACHE_MANIFEST_SHA256}"
test "$(sha256sum "${PREDICTION_DIR}/PREDICTION_MANIFEST.json" | cut -d' ' -f1)" = "${PREDICTION_MANIFEST_SHA256}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

for split in antigen_fold temporal; do
  output_name="chimera-case-study-${split}-${CASE_STUDY_VERSION}"
  test ! -e "${OUTPUT_DIR}/${output_name}"
  docker run --rm --entrypoint sh \
    -e PYTHONPATH=/work/src \
    -v "${SOURCE_DIR}:/work:ro" \
    -v "${CACHE_DIR}:/cache:ro" \
    -v "${PREDICTION_DIR}:/predictions:ro" \
    -v "${OUTPUT_DIR}:/outputs" \
    -w /work "${EVAL_IMAGE}" -lc \
    "python scripts/analyze_chimera_epitope_cases.py \
      --cache-root /cache \
      --expected-cache-manifest-sha256 ${CACHE_MANIFEST_SHA256} \
      --prediction-root /predictions \
      --expected-prediction-manifest-sha256 ${PREDICTION_MANIFEST_SHA256} \
      --split ${split}.test \
      --bootstrap-replicates 5000 \
      --seed 20260730 \
      --output-root /outputs/${output_name}" \
    2>&1 | tee "${LOG_DIR}/${output_name}.log"
done

sha256sum \
  "${OUTPUT_DIR}/chimera-case-study-antigen_fold-${CASE_STUDY_VERSION}/CASE_STUDY.md" \
  "${OUTPUT_DIR}/chimera-case-study-antigen_fold-${CASE_STUDY_VERSION}/CASE_STUDY.json" \
  "${OUTPUT_DIR}/chimera-case-study-antigen_fold-${CASE_STUDY_VERSION}/PER_COMPLEX_CASES.csv" \
  "${OUTPUT_DIR}/chimera-case-study-temporal-${CASE_STUDY_VERSION}/CASE_STUDY.md" \
  "${OUTPUT_DIR}/chimera-case-study-temporal-${CASE_STUDY_VERSION}/CASE_STUDY.json" \
  "${OUTPUT_DIR}/chimera-case-study-temporal-${CASE_STUDY_VERSION}/PER_COMPLEX_CASES.csv" \
  | tee "${LOG_DIR}/chimera-case-study-${CASE_STUDY_VERSION}-final-sha256.txt"
