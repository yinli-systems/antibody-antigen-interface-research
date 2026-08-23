#!/usr/bin/env bash
#SBATCH -J abpairfm-a1-dev-successor
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-dev-successor-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-a1-dev-successor-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly RETRIEVAL_PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source_candidates/fa49bde
readonly CONTACT_PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source_candidates/1f243bb
readonly PROMOTION_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a1_length_band_promotion_v14
readonly PROMOTION_RESULT=${PROMOTION_ROOT}/RESULT.json
readonly DCP_ROOT=${PROMOTION_ROOT}/checkpoints
readonly DCP_GENERATION=stage-a1-000000
readonly RETRIEVAL_PROTOCOL_SHA256=5469ad4cd27ef6bb2c788b55d8c19590feb64bb390fa98f217d9c7229e6edb68
readonly RETRIEVAL_SOURCE_SHA256=7f9da9637b091e7bff82b9dd084c2a72168d4571c758abba493313af9fb81c02
readonly CONTACT_PROTOCOL_SHA256=ec69f3872068b4161babf047708b2edd349005c22fec894c20ce167b66d488ed
readonly CONTACT_SOURCE_SHA256=03028735f9420ed9180d59e4b5dc690f918e9802c03daea87840ca1522ea5d09
readonly RETRIEVAL_OUTPUT=/ssd/scxi253/model_deployment/results/evaluation/abpairfm_stage_a1_development_retrieval_v1
readonly CONTACT_OUTPUT=/ssd/scxi253/model_deployment/results/evaluation/abpairfm_stage_a1_development_contact_v1

mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if [[ "${#GPU_NAMES[@]}" -ne 1 || "${GPU_NAMES[0]}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden development successor GPU allocation" >&2
  exit 2
fi

for path in \
  "${PROMOTION_RESULT}" \
  "${RETRIEVAL_PROJECT}/scripts/submit_abpairfm_stage_a1_frozen_development_retrieval.sbatch" \
  "${CONTACT_PROJECT}/scripts/submit_abpairfm_stage_a1_frozen_development_contact.sbatch"; do
  test -f "${path}"
done
test ! -e "${RETRIEVAL_OUTPUT}"
test ! -e "${CONTACT_OUTPUT}"

readonly DCP_CONTENT_SHA256=$(
  PYTHONPATH="${CONTACT_PROJECT}/src" "${PYTHON}" - \
    "${PROMOTION_RESULT}" "${DCP_ROOT}" "${DCP_GENERATION}" <<'PY'
import json
from pathlib import Path
import sys

from interaction_first.frontier_fsdp2 import verify_dcp_generation

result_path = Path(sys.argv[1])
dcp_root = Path(sys.argv[2])
generation = sys.argv[3]
result = json.loads(result_path.read_text(encoding="utf-8"))
receipt = verify_dcp_generation(dcp_root, generation)
expected_protocol = "9ee2b61d1fc7ee9107c509194b30efcce1a2028578d30c15fc0bc082b15e7ef3"
expected_source = "4b0906eca32564566282e553a642974d2da7d96f2644b27d51b8183f19cdd52b"
content_sha256 = result.get("checkpoint", {}).get("content_sha256")
if (
    result.get("status")
    != "PASS_STAGE_A1_LENGTH_BAND_PROMOTION_CHECKPOINT_VERIFIED"
    or result.get("test_access") is not False
    or result.get("source_sha256") != expected_source
    or result.get("protocol_sha256") != expected_protocol
    or result.get("checkpoint", {}).get("generation") != generation
    or receipt.get("generation") != generation
    or receipt.get("content_sha256") != content_sha256
    or receipt.get("provenance_hashes", {}).get("protocol") != expected_protocol
    or receipt.get("provenance_hashes", {}).get("source") != expected_source
    or receipt.get("training_state", {}).get("stage_a1_updates") != 200
    or receipt.get("training_state", {}).get("production_training") is not True
    or receipt.get("training_state", {}).get("test_access") is not False
):
    raise SystemExit("Stage-A1 promotion result does not authorize development evaluation")
print(content_sha256)
PY
)
if [[ ! "${DCP_CONTENT_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Stage-A1 DCP content SHA-256 is invalid" >&2
  exit 2
fi

readonly RETRIEVAL_JOB=$(
  sbatch --parsable \
    --export=ALL,ABPAIRFM_STAGE_A1_EVAL_PROJECT="${RETRIEVAL_PROJECT}",ABPAIRFM_STAGE_A1_EVAL_PROTOCOL_SHA256="${RETRIEVAL_PROTOCOL_SHA256}",ABPAIRFM_STAGE_A1_EVAL_SOURCE_SHA256="${RETRIEVAL_SOURCE_SHA256}",ABPAIRFM_STAGE_A1_EVAL_DCP_ROOT="${DCP_ROOT}",ABPAIRFM_STAGE_A1_EVAL_DCP_GENERATION="${DCP_GENERATION}",ABPAIRFM_STAGE_A1_EVAL_DCP_CONTENT_SHA256="${DCP_CONTENT_SHA256}",ABPAIRFM_STAGE_A1_EVAL_PROMOTION_RESULT="${PROMOTION_RESULT}",ABPAIRFM_STAGE_A1_EVAL_OUTPUT_ROOT="${RETRIEVAL_OUTPUT}" \
    "${RETRIEVAL_PROJECT}/scripts/submit_abpairfm_stage_a1_frozen_development_retrieval.sbatch"
)
readonly CONTACT_JOB=$(
  sbatch --parsable \
    --export=ALL,ABPAIRFM_STAGE_A1_CONTACT_PROJECT="${CONTACT_PROJECT}",ABPAIRFM_STAGE_A1_CONTACT_PROTOCOL_SHA256="${CONTACT_PROTOCOL_SHA256}",ABPAIRFM_STAGE_A1_CONTACT_SOURCE_SHA256="${CONTACT_SOURCE_SHA256}",ABPAIRFM_STAGE_A1_CONTACT_DCP_ROOT="${DCP_ROOT}",ABPAIRFM_STAGE_A1_CONTACT_DCP_GENERATION="${DCP_GENERATION}",ABPAIRFM_STAGE_A1_CONTACT_DCP_CONTENT_SHA256="${DCP_CONTENT_SHA256}",ABPAIRFM_STAGE_A1_CONTACT_PROMOTION_RESULT="${PROMOTION_RESULT}",ABPAIRFM_STAGE_A1_CONTACT_OUTPUT_ROOT="${CONTACT_OUTPUT}" \
    "${CONTACT_PROJECT}/scripts/submit_abpairfm_stage_a1_frozen_development_contact.sbatch"
)

printf 'retrieval_job=%s contact_job=%s dcp_content_sha256=%s test_access=false\n' \
  "${RETRIEVAL_JOB}" "${CONTACT_JOB}" "${DCP_CONTENT_SHA256}"
