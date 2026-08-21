#!/usr/bin/env bash
#SBATCH -J abpairfm-string-filter
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-filter-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-filter-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly INPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/string_v12/protein.physical.links.full.v12.0.txt.gz
readonly INPUT_SHA256=1d5d16bd8aeb67c9977948a0e56cc27cca2065ad981dab4586330f38d77ff234
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly OUTPUT="${OUTPUT_ROOT}/direct_physical_combined700_v1.tsv.zst"
readonly RECEIPT="${OUTPUT_ROOT}/direct_physical_combined700_v1.receipt.json"
readonly FILTER_SCRIPT="${PROJECT}/scripts/filter_string_v12_direct_physical.sbatch"

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ -e "${OUTPUT}" || -e "${RECEIPT}" ]]; then
  echo "Refusing to overwrite STRING filter evidence" >&2
  exit 2
fi
readonly STAGING=$(mktemp "${OUTPUT_ROOT}/.direct_physical_combined700_v1.XXXXXX.tsv.zst")
readonly STATS=$(mktemp "${OUTPUT_ROOT}/.direct_physical_combined700_v1.XXXXXX.stats")
cleanup_filter() {
  [[ ! -e "${STAGING}" ]] || rm -- "${STAGING}"
  [[ ! -e "${STATS}" ]] || rm -- "${STATS}"
}
trap cleanup_filter EXIT

printf '%s  %s\n' "${INPUT_SHA256}" "${INPUT}" | sha256sum --check --strict
export LC_ALL=C
gzip -cd "${INPUT}" | \
  awk -v stats="${STATS}" '
    NR == 1 {
      expected = "protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score"
      if ($0 != expected) {
        print "Unexpected STRING header" > "/dev/stderr"
        exit 3
      }
      print $0
      next
    }
    {
      source_rows++
      if (NF != 10) {
        malformed_rows++
        next
      }
      if ($10 >= 700) {
        high_confidence_rows++
        if ($4 > 0 || $6 > 0) {
          direct_rows++
          print $0
        } else if ($5 > 0 || $7 > 0) {
          transferred_only_rows++
        }
      }
    }
    END {
      print "combined_score_ge_700_rows=" high_confidence_rows > stats
      print "direct_evidence_kept_rows=" direct_rows >> stats
      print "malformed_rows=" malformed_rows + 0 >> stats
      print "source_data_rows=" source_rows >> stats
      print "transferred_only_ge_700_rows=" transferred_only_rows + 0 >> stats
      if (malformed_rows > 0 || direct_rows == 0) {
        exit 4
      }
    }
  ' | zstd -T4 -3 --check --force -o "${STAGING}"

zstd --test "${STAGING}"
chmod 0444 "${STAGING}"
ln "${STAGING}" "${OUTPUT}"
rm -- "${STAGING}"
readonly FILTER_SCRIPT_SHA256=$(sha256sum "${FILTER_SCRIPT}" | cut -d ' ' -f 1)
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_direct_filter.py" \
  --input "${INPUT}" \
  --input-sha256 "${INPUT_SHA256}" \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}" \
  --stats "${STATS}" \
  --filter-script-sha256 "${FILTER_SCRIPT_SHA256}" \
  --job-id "${SLURM_JOB_ID}" \
  --gpu-name "${GPU_NAME}"
