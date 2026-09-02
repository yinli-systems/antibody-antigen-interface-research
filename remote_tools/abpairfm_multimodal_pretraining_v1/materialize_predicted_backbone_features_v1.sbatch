#!/usr/bin/env bash
#SBATCH -J abpairfm-mm-backbone-features
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/backbone-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm_pinder_multimodal_v1/backbone-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly SOURCE=${ROOT}/remote_tools/abpairfm_multimodal_pretraining_v1
readonly OUTPUT=${ROOT}/results/pretraining/abpairfm_pinder_multimodal_v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly MATERIALIZER=${SOURCE}/materialize_predicted_backbone_features_v1.py
readonly MATERIALIZER_SHA=df35ae14fda7b648e2a19d12c2a20207833976558d615f19d85b3c99e06ec0e9
readonly CACHE_RESULT=${OUTPUT}/PINDER_PREDICTED_STRUCTURE_CACHE_RESULT_V1.json
readonly CACHE_ROOT=${OUTPUT}/predicted_structure_cache_v1
readonly STRUCTURE_MANIFEST=${CACHE_ROOT}/PREDICTED_STRUCTURE_CACHE_MANIFEST_V1.tsv
readonly FEATURE_BINARY=${OUTPUT}/PINDER_PREDICTED_BACKBONE_FEATURES_V1.f32
readonly FEATURE_INDEX=${OUTPUT}/PINDER_PREDICTED_BACKBONE_FEATURES_INDEX_V1.tsv
readonly RESULT=${OUTPUT}/PINDER_PREDICTED_BACKBONE_FEATURES_RESULT_V1.json

test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA GeForce RTX 4090"
test "$(sha256sum "${MATERIALIZER}" | cut -d' ' -f1)" = "${MATERIALIZER_SHA}"
test ! -e "${FEATURE_BINARY}"
test ! -e "${FEATURE_INDEX}"
test ! -e "${RESULT}"

readonly STRUCTURE_MANIFEST_SHA=$(sha256sum "${STRUCTURE_MANIFEST}" | cut -d' ' -f1)
"${PYTHON}" - "${CACHE_RESULT}" "${STRUCTURE_MANIFEST_SHA}" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="ascii"))
if result.get("status") != "PASS_AUTHENTICATED_PREDICTED_STRUCTURE_CACHE":
    raise SystemExit("predicted-structure cache did not pass")
if result.get("outputs", {}).get("manifest_sha256") != sys.argv[2]:
    raise SystemExit("predicted-structure manifest SHA differs from cache result")
PY

"${PYTHON}" "${MATERIALIZER}" \
  --structure-root "${CACHE_ROOT}" \
  --structure-manifest "${STRUCTURE_MANIFEST}" \
  --structure-manifest-sha256 "${STRUCTURE_MANIFEST_SHA}" \
  --output-features "${FEATURE_BINARY}" \
  --output-index "${FEATURE_INDEX}" \
  --output-result "${RESULT}"
sha256sum "${FEATURE_BINARY}" "${FEATURE_INDEX}" "${RESULT}"
