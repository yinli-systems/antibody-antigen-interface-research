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
readonly TRAIN_PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_stage_a0_4gpu_production_v1/source
readonly CONTINUATION_ROOT=/ssd/scxi253/model_deployment/abpairfm_3b_unique25m_continuation_v1
readonly PROBE_PROJECT=${ABPAIRFM_PROBE_PROJECT_ROOT:-/ssd/scxi253/model_deployment/abpairfm_3b_v1/source_candidates/2205f62}
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/stage_a0_string25m_v1
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_stage_a0_string25m_v1
readonly RETENTION_SCRIPT=${CONTINUATION_ROOT}/scripts/prune_abpairfm_3b_checkpoints.py
readonly RETENTION_SCRIPT_SHA256=1592d09ca54672a515a7c12a827a267ce74c716927318bf8ce916d322a1af3e2
readonly RETENTION_AUDIT_ROOT=${OUTPUT_ROOT}/audits/checkpoint_retention_v1
readonly PLAN=${CONTINUATION_ROOT}/ABPAIRFM_3B_UNIQUE25M_CONTINUATION_PLAN_V1.json
readonly PLAN_SHA256=1c246cabf4a64b9db3effbe65a3dbb2681754cf416d5a893f2976316e86aa877
readonly SOURCE_MANIFEST=${CONTINUATION_ROOT}/ABPAIRFM_3B_UNIQUE25M_CONTINUATION_SOURCE_MANIFEST_V2.json
readonly SOURCE_RECEIPT=${CONTINUATION_ROOT}/ABPAIRFM_3B_UNIQUE25M_CONTINUATION_SOURCE_MANIFEST_V2_RECEIPT.json
readonly PROTOCOL_SHA256=a783f3b20516fbb2c85dba27833d3125683184ae2be608dfcfdbf91e58a3747b
readonly SOURCE_SHA256=8b94d5fa308c9845a5567b80b9dab6d4a35e298a91c2a807521774896be946f8
readonly TARGET_RESIDUE_TOKENS=5000000000
readonly RECORDS_PER_SEGMENT=22400
readonly MAXIMUM_UNIQUE_PAIRS=25000000
readonly MAX_UNIQUE_SEGMENTS=$((MAXIMUM_UNIQUE_PAIRS / RECORDS_PER_SEGMENT))
readonly EXCLUDED_TRAINING_NODE=wqd10naf05g3
readonly COMPLETED_INDEX=${ABPAIRFM_COMPLETED_SEGMENT_INDEX:?missing completed segment index}
readonly NEXT_INDEX=$((COMPLETED_INDEX + 1))
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
for path in "${PLAN}" "${SOURCE_MANIFEST}" "${SOURCE_RECEIPT}" "${RESULT}" "${NEXT_RECEIPT}"; do
  test -f "${path}"
done
if [[ "$(sha256sum "${RETENTION_SCRIPT}" | awk '{print $1}')" != "${RETENTION_SCRIPT_SHA256}" ]]; then
  echo "Checkpoint-retention source SHA-256 differs" >&2
  exit 2
fi

mapfile -t CHECKPOINT_ANCHORS < <(
  "${PYTHON}" - \
    "${PLAN}" "${PLAN_SHA256}" "${SOURCE_MANIFEST}" "${SOURCE_RECEIPT}" \
    "${RESULT}" "${NEXT_RECEIPT}" "${COMPLETED_INDEX}" "${NEXT_INDEX}" \
    "${PROTOCOL_SHA256}" "${SOURCE_SHA256}" "${TARGET_RESIDUE_TOKENS}" \
    "${RECORDS_PER_SEGMENT}" "${MAX_UNIQUE_SEGMENTS}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

(
    plan_path,
    expected_plan_sha,
    source_manifest_path,
    source_receipt_path,
    result_path,
    next_receipt_path,
) = map(Path, sys.argv[1:7])
completed_index, next_index = map(int, sys.argv[7:9])
protocol_sha, source_sha = sys.argv[9:11]
target_tokens, records_per_segment, maximum_segments = map(int, sys.argv[11:14])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if sha256_file(plan_path) != str(expected_plan_sha):
    raise SystemExit("Unique25M continuation plan SHA-256 differs")
plan = json.loads(plan_path.read_text(encoding="utf-8"))
if (
    plan.get("format_version") != "abpairfm_3b_unique25m_continuation_plan_v1"
    or plan.get("status") != "FROZEN_BEFORE_SEGMENT_000112_SUBMISSION"
    or plan.get("authorization", {}).get("continue_production_stage_a0_pair_mlm") is not True
    or plan.get("authorization", {}).get("repeat_previously_consumed_pair_slices") is not False
    or plan.get("test_access") is not False
    or plan.get("continuation", {}).get("maximum_whole_unique_segments") != maximum_segments
    or plan.get("continuation", {}).get("target_consumed_residue_tokens") != target_tokens
):
    raise SystemExit("Unique25M continuation plan contract differs")

source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
if (
    source_receipt.get("format_version") != "abpairfm_3b_unique25m_source_manifest_receipt_v2"
    or source_receipt.get("status") != "PASS_SOURCE_FROZEN"
    or source_receipt.get("source_manifest_sha256") != sha256_file(source_manifest_path)
    or source_receipt.get("scientific_protocol_changed") is not False
    or source_receipt.get("test_access") is not False
):
    raise SystemExit("Unique25M source-manifest receipt differs")
source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
if (
    source_manifest.get("format_version") != "abpairfm_3b_unique25m_source_manifest_v2"
    or source_manifest.get("status") != "FROZEN_BEFORE_SEGMENT_000117_RECOVERY"
    or source_manifest.get("scientific_protocol_changed") is not False
    or source_manifest.get("test_access") is not False
):
    raise SystemExit("Unique25M source manifest contract differs")
for entry in source_manifest.get("files", []):
    path = Path(entry["path"])
    if sha256_file(path) != entry["sha256"]:
        raise SystemExit(f"Frozen continuation source differs: {path}")

result = json.loads(result_path.read_text(encoding="utf-8"))
if (
    result.get("format_version") != "abpairfm_3b_stage_a0_segment_result_v1"
    or result.get("status") != "PASS_CHECKPOINT_VERIFIED"
    or result.get("segment_index") != completed_index
    or result.get("campaign_protocol_sha256") != protocol_sha
    or result.get("source_sha256") != source_sha
    or result.get("test_access") is not False
    or result.get("training_state", {}).get("next_segment_index") != next_index
):
    raise SystemExit("Completed Stage-A0 result does not authorize a successor")
checkpoint = result.get("checkpoint", {})
root = Path(checkpoint.get("root", ""))
generation = checkpoint.get("generation")
content_sha = checkpoint.get("content_sha256")
checkpoint_receipt = json.loads((root / str(generation) / "receipt.json").read_text(encoding="utf-8"))
if (
    checkpoint_receipt.get("content_sha256") != content_sha
    or checkpoint_receipt.get("training_state") != result.get("training_state")
    or checkpoint_receipt.get("provenance_hashes", {}).get("campaign_protocol") != protocol_sha
    or checkpoint_receipt.get("provenance_hashes", {}).get("training_source") != source_sha
):
    raise SystemExit("Immutable DCP receipt differs from the passing result")

next_receipt = json.loads(next_receipt_path.read_text(encoding="utf-8"))
segment = next_receipt.get("segment", {})
expected_start = next_index * records_per_segment
if (
    next_receipt.get("format_version") != "abpairfm_3b_stage_a0_segment_manifest_v1"
    or next_receipt.get("status") != "PRODUCTION_STAGE_A0_SEGMENT_READY"
    or next_receipt.get("test_access") is not False
    or segment.get("index") != next_index
    or segment.get("selection_start") != expected_start
    or segment.get("selection_stop_exclusive") != expected_start + records_per_segment
    or next_receipt.get("counts", {}).get("manifest_records") != records_per_segment
    or next_receipt.get("counts", {}).get("unique_biological_pairs") != records_per_segment
    or next_receipt.get("inputs", {}).get("protocol_sha256") != protocol_sha
):
    raise SystemExit("Next unique Stage-A0 manifest receipt differs")
manifest_path = next_receipt_path.with_name(next_receipt["output"]["name"])
if sha256_file(manifest_path) != next_receipt["output"]["sha256"]:
    raise SystemExit("Next unique Stage-A0 manifest bytes differ")

consumed = int(result.get("training_state", {}).get("consumed_residue_tokens", -1))
print(root)
print(generation)
print(content_sha)
print("true" if consumed >= target_tokens else "false")
PY
)
if [[ "${#CHECKPOINT_ANCHORS[@]}" -ne 4 ]]; then
  echo "Successor anchor extraction returned an unexpected field count" >&2
  exit 2
fi

"${PYTHON}" "${PROBE_PROJECT}/scripts/maybe_submit_abpairfm_checkpoint_ladder_probe.py" \
  --project-root "${PROBE_PROJECT}" \
  --result "${RESULT}" \
  --protocol "${PROBE_PROJECT}/protocols/abpairfm_checkpoint_transfer_probe_v1.json" \
  --expected-protocol-sha256 57d5049b67795079b4b20eb37ed18b8026fd35f05e5a06991dcab732a96b1809 \
  --submission-root /ssd/scxi253/model_deployment/results/evaluation/abpairfm_checkpoint_transfer_probe_v1/submissions

"${PYTHON}" "${RETENTION_SCRIPT}" \
  --root "${OUTPUT_ROOT}" \
  --keep-full 2 \
  --audit-root "${RETENTION_AUDIT_ROOT}" \
  --execute

if [[ "${CHECKPOINT_ANCHORS[3]}" == "true" ]]; then
  printf 'status=PASS_FROZEN_TOKEN_TARGET_REACHED completed_segment=%s consumed_target=%s\n' \
    "${COMPLETED_SEGMENT}" "${TARGET_RESIDUE_TOKENS}"
  exit 0
fi

readonly NEXT_JOB=$(
  sbatch --parsable \
    --exclude="${EXCLUDED_TRAINING_NODE}" \
    --export=ALL,ABPAIRFM_SEGMENT_INDEX="${NEXT_INDEX}",ABPAIRFM_INPUT_CHECKPOINT_ROOT="${CHECKPOINT_ANCHORS[0]}",ABPAIRFM_INPUT_GENERATION="${CHECKPOINT_ANCHORS[1]}",ABPAIRFM_EXPECTED_INPUT_CONTENT_SHA256="${CHECKPOINT_ANCHORS[2]}" \
    "${TRAIN_PROJECT}/scripts/submit_abpairfm_3b_stage_a0_segment.sbatch"
)

if [[ $((NEXT_INDEX + 1)) -lt "${MAX_UNIQUE_SEGMENTS}" ]]; then
  readonly FOLLOWING_MANIFEST_JOB=$(
    sbatch --parsable \
      --export=ALL,ABPAIRFM_SEGMENT_INDEX="$((NEXT_INDEX + 1))" \
      "${TRAIN_PROJECT}/scripts/build_abpairfm_3b_stage_a0_segment.sbatch"
  )
  readonly WATCHDOG_JOB=$(
    sbatch --parsable \
      --dependency=afterok:"${NEXT_JOB}":"${FOLLOWING_MANIFEST_JOB}" \
      --export=ALL,ABPAIRFM_COMPLETED_SEGMENT_INDEX="${NEXT_INDEX}" \
      "${CONTINUATION_ROOT}/continue_abpairfm_3b_unique25m_v1.sbatch"
  )
  printf 'next_job=%s following_manifest_job=%s watchdog_job=%s unique_manifest_frontier=%s\n' \
    "${NEXT_JOB}" "${FOLLOWING_MANIFEST_JOB}" "${WATCHDOG_JOB}" "${MAX_UNIQUE_SEGMENTS}"
else
  printf 'next_job=%s unique_manifest_frontier=%s\n' "${NEXT_JOB}" "${MAX_UNIQUE_SEGMENTS}"
fi
