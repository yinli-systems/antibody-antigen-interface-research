#!/usr/bin/env bash
set -euo pipefail

: "${FRAMECONTACT_FOLDSEEK:?Set FRAMECONTACT_FOLDSEEK}"
: "${FRAMECONTACT_CROSS_PREPARATION_ROOT:?Set FRAMECONTACT_CROSS_PREPARATION_ROOT}"
: "${FRAMECONTACT_CROSS_FOLDSEEK_OUTPUT_ROOT:?Set FRAMECONTACT_CROSS_FOLDSEEK_OUTPUT_ROOT}"
: "${FRAMECONTACT_PYTHON_BIN:?Set FRAMECONTACT_PYTHON_BIN}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
preparation_root="${FRAMECONTACT_CROSS_PREPARATION_ROOT}"
output_root="${FRAMECONTACT_CROSS_FOLDSEEK_OUTPUT_ROOT}"
threads="${FRAMECONTACT_FOLDSEEK_THREADS:-16}"
if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite cross Foldseek output: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}/tmp"
raw="${output_root}/foldseek.raw.tsv"
normalized="${output_root}/foldseek.normalized.tsv"
command_path="${output_root}/COMMAND.txt"
receipt="${output_root}/CROSS_FOLDSEEK_RECEIPT.json"
command=(
  "${FRAMECONTACT_FOLDSEEK}" easy-search
  "${preparation_root}/queries" "${preparation_root}/targets"
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
preparation_receipt="${preparation_root}/PREPARATION_RECEIPT.json"
preparation_sha="$(sha256sum "${preparation_receipt}" | awk '{print $1}')"
foldseek_version="$("${FRAMECONTACT_FOLDSEEK}" version)"
PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${FRAMECONTACT_PYTHON_BIN}" \
  "${repository_root}/scripts/finalize_asep_sabdab2_cross_foldseek.py" \
  --preparation-root "${preparation_root}" \
  --preparation-receipt-sha256 "${preparation_sha}" \
  --raw-alignments "${raw}" \
  --raw-alignments-sha256 "${raw_sha}" \
  --command "${command_path}" \
  --foldseek-version "${foldseek_version}" \
  --normalized-output "${normalized}" \
  --receipt-output "${receipt}"
sha256sum "${raw}" "${normalized}" "${receipt}" "${command_path}" \
  > "${output_root}/SHA256SUMS.txt"
