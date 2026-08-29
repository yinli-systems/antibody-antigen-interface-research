#!/usr/bin/env bash
#SBATCH -J abpairfm-3b-a0-successor
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-successor-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-3b-a0-successor-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_stage_a0_4gpu_production_v1/source
readonly PROBE_PROJECT=${ABPAIRFM_PROBE_PROJECT_ROOT:-/ssd/scxi253/model_deployment/abpairfm_3b_v1/source_candidates/2205f62}
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a0_string25m_v1
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1
readonly PROTOCOL_SHA256=a783f3b20516fbb2c85dba27833d3125683184ae2be608dfcfdbf91e58a3747b
readonly SOURCE_SHA256=8b94d5fa308c9845a5567b80b9dab6d4a35e298a91c2a807521774896be946f8
readonly EXCLUDED_TRAINING_NODE=wqd10naf05g3
readonly COMPLETED_INDEX=${ABPAIRFM_COMPLETED_SEGMENT_INDEX:?missing completed segment index}
readonly NEXT_INDEX=$((COMPLETED_INDEX + 1))
readonly MAX_UNIQUE_SEGMENTS=112
readonly COMPLETED_SEGMENT=$(printf 'segment-%06d' "${COMPLETED_INDEX}")
readonly NEXT_SEGMENT=$(printf 'segment-%06d' "${NEXT_INDEX}")
readonly RESULT=${OUTPUT_ROOT}/results/${COMPLETED_SEGMENT}.json
readonly NEXT_RECEIPT=${DATA_ROOT}/string_seed1701_${NEXT_SEGMENT}_22400_v1.receipt.json
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden successor allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
if [[ "${COMPLETED_INDEX}" -lt 0 || "${NEXT_INDEX}" -ge "${MAX_UNIQUE_SEGMENTS}" ]]; then
  echo "Unique-pair Stage-A0 successor frontier reached or index invalid" >&2
  exit 2
fi
for path in "${RESULT}" "${NEXT_RECEIPT}"; do
  test -f "${path}"
done

mapfile -t CHECKPOINT_ANCHORS < <(
  "${PYTHON}" - "${RESULT}" "${NEXT_RECEIPT}" "${COMPLETED_INDEX}" "${NEXT_INDEX}" "${PROTOCOL_SHA256}" "${SOURCE_SHA256}" <<'PY'
import json
from pathlib import Path
import sys

result_path, next_receipt_path = map(Path, sys.argv[1:3])
completed_index, next_index = map(int, sys.argv[3:5])
protocol_sha256, source_sha256 = sys.argv[5:7]
result = json.loads(result_path.read_text(encoding="utf-8"))
if (
    result.get("format_version") != "abpairfm_3b_stage_a0_segment_result_v1"
    or result.get("status") != "PASS_CHECKPOINT_VERIFIED"
    or result.get("segment_index") != completed_index
    or result.get("campaign_protocol_sha256") != protocol_sha256
    or result.get("source_sha256") != source_sha256
    or result.get("test_access") is not False
    or result.get("training_state", {}).get("next_segment_index") != next_index
):
    raise SystemExit("Completed Stage-A0 result does not authorize a successor")
checkpoint = result.get("checkpoint", {})
root = Path(checkpoint.get("root", ""))
generation = checkpoint.get("generation")
content_sha256 = checkpoint.get("content_sha256")
receipt_path = root / str(generation) / "receipt.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("content_sha256") != content_sha256
    or receipt.get("training_state") != result.get("training_state")
    or receipt.get("provenance_hashes", {}).get("campaign_protocol") != protocol_sha256
    or receipt.get("provenance_hashes", {}).get("training_source") != source_sha256
):
    raise SystemExit("Immutable DCP receipt differs from the passing result")
next_receipt = json.loads(next_receipt_path.read_text(encoding="utf-8"))
if (
    next_receipt.get("format_version") != "abpairfm_3b_stage_a0_segment_manifest_v1"
    or next_receipt.get("status") != "PRODUCTION_STAGE_A0_SEGMENT_READY"
    or next_receipt.get("segment", {}).get("index") != next_index
    or next_receipt.get("inputs", {}).get("protocol_sha256") != protocol_sha256
    or next_receipt.get("test_access") is not False
):
    raise SystemExit("Next Stage-A0 manifest receipt differs")
print(root)
print(generation)
print(content_sha256)
PY
)
if [[ "${#CHECKPOINT_ANCHORS[@]}" -ne 3 ]]; then
  echo "Successor anchor extraction returned an unexpected field count" >&2
  exit 2
fi

"${PYTHON}" "${PROBE_PROJECT}/scripts/maybe_submit_abpairfm_checkpoint_ladder_probe.py" \
  --project-root "${PROBE_PROJECT}" \
  --result "${RESULT}" \
  --protocol "${PROBE_PROJECT}/protocols/abpairfm_checkpoint_transfer_probe_v1.json" \
  --expected-protocol-sha256 57d5049b67795079b4b20eb37ed18b8026fd35f05e5a06991dcab732a96b1809 \
  --submission-root /ssd/scxi253/model_deployment/results/evaluation/abpairfm_checkpoint_transfer_probe_v1/submissions

readonly NEXT_JOB=$(
  sbatch --parsable \
    --exclude="${EXCLUDED_TRAINING_NODE}" \
    --export=ALL,ABPAIRFM_SEGMENT_INDEX="${NEXT_INDEX}",ABPAIRFM_INPUT_CHECKPOINT_ROOT="${CHECKPOINT_ANCHORS[0]}",ABPAIRFM_INPUT_GENERATION="${CHECKPOINT_ANCHORS[1]}",ABPAIRFM_EXPECTED_INPUT_CONTENT_SHA256="${CHECKPOINT_ANCHORS[2]}" \
    "${PROJECT}/scripts/submit_abpairfm_3b_stage_a0_segment.sbatch"
)

if [[ $((NEXT_INDEX + 1)) -lt "${MAX_UNIQUE_SEGMENTS}" ]]; then
  readonly FOLLOWING_MANIFEST_JOB=$(
    sbatch --parsable \
      --export=ALL,ABPAIRFM_SEGMENT_INDEX="$((NEXT_INDEX + 1))" \
      "${PROJECT}/scripts/build_abpairfm_3b_stage_a0_segment.sbatch"
  )
  readonly WATCHDOG_JOB=$(
    sbatch --parsable \
      --dependency=afterok:"${NEXT_JOB}":"${FOLLOWING_MANIFEST_JOB}" \
      --export=ALL,ABPAIRFM_COMPLETED_SEGMENT_INDEX="${NEXT_INDEX}" \
      "${PROJECT}/scripts/continue_abpairfm_3b_stage_a0.sbatch"
  )
  printf 'next_job=%s following_manifest_job=%s watchdog_job=%s excluded_training_node=%s\n' \
    "${NEXT_JOB}" "${FOLLOWING_MANIFEST_JOB}" "${WATCHDOG_JOB}" "${EXCLUDED_TRAINING_NODE}"
else
  printf 'next_job=%s unique_manifest_frontier=%s excluded_training_node=%s\n' \
    "${NEXT_JOB}" "${MAX_UNIQUE_SEGMENTS}" "${EXCLUDED_TRAINING_NODE}"
fi
