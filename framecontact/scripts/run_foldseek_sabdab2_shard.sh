#!/usr/bin/env bash
set -euo pipefail

: "${FRAMECONTACT_FOLDSEEK:?Set FRAMECONTACT_FOLDSEEK}"
: "${FRAMECONTACT_ANTIGEN_EXPORT_ROOT:?Set FRAMECONTACT_ANTIGEN_EXPORT_ROOT}"
: "${FRAMECONTACT_FOLDSEEK_SHARD_PLAN_ROOT:?Set FRAMECONTACT_FOLDSEEK_SHARD_PLAN_ROOT}"
: "${FRAMECONTACT_FOLDSEEK_SHARD_OUTPUT_ROOT:?Set FRAMECONTACT_FOLDSEEK_SHARD_OUTPUT_ROOT}"
: "${FRAMECONTACT_FOLDSEEK_SHARD_INDEX:?Set FRAMECONTACT_FOLDSEEK_SHARD_INDEX}"
: "${FRAMECONTACT_PYTHON_BIN:?Set FRAMECONTACT_PYTHON_BIN}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shard_name="$(printf 'shard-%03d' "${FRAMECONTACT_FOLDSEEK_SHARD_INDEX}")"
query_structures="${FRAMECONTACT_FOLDSEEK_SHARD_PLAN_ROOT}/${shard_name}/structures"
target_structures="${FRAMECONTACT_ANTIGEN_EXPORT_ROOT}/structures"
plan="${FRAMECONTACT_FOLDSEEK_SHARD_PLAN_ROOT}/FOLDSEEK_SHARD_PLAN.json"
output_root="${FRAMECONTACT_FOLDSEEK_SHARD_OUTPUT_ROOT}/${shard_name}"
threads="${FRAMECONTACT_FOLDSEEK_THREADS:-6}"

if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite Foldseek shard output: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}/tmp"
raw="${output_root}/foldseek.raw.tsv"
command_path="${output_root}/COMMAND.txt"
command=(
  "${FRAMECONTACT_FOLDSEEK}" easy-search "${query_structures}" "${target_structures}"
  "${raw}" "${output_root}/tmp"
  --alignment-type 1
  --exhaustive-search 1
  -e inf
  --max-seqs 100000
  --threads "${threads}"
  --format-output query,target,qtmscore,ttmscore
)
printf '%q ' "${command[@]}" > "${command_path}"
printf '\n' >> "${command_path}"
"${command[@]}" > "${output_root}/foldseek.stdout.log" \
  2> "${output_root}/foldseek.stderr.log"

raw_sha="$(sha256sum "${raw}" | awk '{print $1}')"
plan_sha="$(sha256sum "${plan}" | awk '{print $1}')"
foldseek_version="$("${FRAMECONTACT_FOLDSEEK}" version)"
PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${FRAMECONTACT_PYTHON_BIN}" "${repository_root}/scripts/finalize_foldseek_shard.py" \
  --plan "${plan}" \
  --plan-sha256 "${plan_sha}" \
  --shard-index "${FRAMECONTACT_FOLDSEEK_SHARD_INDEX}" \
  --raw "${raw}" \
  --raw-sha256 "${raw_sha}" \
  --command "${command_path}" \
  --receipt "${output_root}/FOLDSEEK_SHARD_RECEIPT.json" \
  --foldseek-version "${foldseek_version}"
sha256sum "${raw}" "${command_path}" \
  "${output_root}/FOLDSEEK_SHARD_RECEIPT.json" > "${output_root}/SHA256SUMS.txt"
