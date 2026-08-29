#!/usr/bin/env bash
set -euo pipefail

: "${FRAMECONTACT_FOLDSEEK:?Set FRAMECONTACT_FOLDSEEK}"
: "${FRAMECONTACT_ANTIGEN_EXPORT_ROOT:?Set FRAMECONTACT_ANTIGEN_EXPORT_ROOT}"
: "${FRAMECONTACT_FOLDSEEK_OUTPUT_ROOT:?Set FRAMECONTACT_FOLDSEEK_OUTPUT_ROOT}"
: "${FRAMECONTACT_PYTHON_BIN:?Set FRAMECONTACT_PYTHON_BIN}"

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
structures="${FRAMECONTACT_ANTIGEN_EXPORT_ROOT}/structures"
manifest="${FRAMECONTACT_ANTIGEN_EXPORT_ROOT}/ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
output_root="${FRAMECONTACT_FOLDSEEK_OUTPUT_ROOT}"
threads="${FRAMECONTACT_FOLDSEEK_THREADS:-16}"
raw="${output_root}/foldseek.raw.tsv"
normalized="${output_root}/foldseek.normalized.tsv"
receipt="${output_root}/FOLDSEEK_RECEIPT.json"
temporary="${output_root}/tmp"

if [[ -e "${output_root}" ]]; then
  echo "Refusing to overwrite Foldseek output root: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}" "${temporary}"

command=(
  "${FRAMECONTACT_FOLDSEEK}" easy-search "${structures}" "${structures}"
  "${raw}" "${temporary}"
  --alignment-type 1
  --exhaustive-search 1
  -e inf
  --max-seqs 100000
  --threads "${threads}"
  --format-output query,target,qtmscore,ttmscore
)
printf '%q ' "${command[@]}" > "${output_root}/COMMAND.txt"
printf '\n' >> "${output_root}/COMMAND.txt"
"${command[@]}" > "${output_root}/foldseek.stdout.log" 2> "${output_root}/foldseek.stderr.log"

raw_sha="$(sha256sum "${raw}" | awk '{print $1}')"
manifest_sha="$(sha256sum "${manifest}" | awk '{print $1}')"
foldseek_version="$("${FRAMECONTACT_FOLDSEEK}" version)"
PYTHONPATH="${repository_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${FRAMECONTACT_PYTHON_BIN}" "${repository_root}/scripts/normalize_foldseek_alignments.py" \
  --raw "${raw}" \
  --raw-sha256 "${raw_sha}" \
  --export-manifest "${manifest}" \
  --export-manifest-sha256 "${manifest_sha}" \
  --output "${normalized}" \
  --receipt "${receipt}" \
  --foldseek-version "${foldseek_version}" \
  --command "$(cat "${output_root}/COMMAND.txt")"

sha256sum "${raw}" "${normalized}" "${receipt}" > "${output_root}/SHA256SUMS.txt"
