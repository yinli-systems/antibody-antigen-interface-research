#!/usr/bin/env bash
#SBATCH -J abpairfm-string-high700
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly INPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/string_v12/protein.physical.links.full.v12.0.txt.gz
readonly INPUT_SHA256=1d5d16bd8aeb67c9977948a0e56cc27cca2065ad981dab4586330f38d77ff234
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly OUTPUT=${OUTPUT_ROOT}/physical_combined700_all_channels_v2.tsv.zst
readonly RECEIPT=${OUTPUT_ROOT}/physical_combined700_all_channels_v2.receipt.json
readonly STAGING=$(mktemp "${OUTPUT_ROOT}/.physical_combined700_all_channels_v2.XXXXXX.tsv.zst")
readonly STATS=$(mktemp "${OUTPUT_ROOT}/.physical_combined700_all_channels_v2.XXXXXX.stats")
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING scale-filter allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" || -e "${RECEIPT}" ]]; then
  echo "Refusing to overwrite STRING scale-filter evidence" >&2
  exit 2
fi
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
        } else if ($5 > 0 || $7 > 0) {
          transferred_rows++
        } else {
          other_rows++
        }
        print $0
      }
    }
    END {
      print "direct_channel_rows=" direct_rows + 0 > stats
      print "high_confidence_rows=" high_confidence_rows + 0 >> stats
      print "malformed_rows=" malformed_rows + 0 >> stats
      print "other_calibrated_channel_rows=" other_rows + 0 >> stats
      print "source_data_rows=" source_rows + 0 >> stats
      print "transferred_channel_rows=" transferred_rows + 0 >> stats
      if (malformed_rows > 0 || high_confidence_rows == 0) {
        exit 4
      }
    }
  ' | zstd -T4 -3 --check --force -o "${STAGING}"

zstd --test "${STAGING}"
chmod 0444 "${STAGING}"
ln "${STAGING}" "${OUTPUT}"
rm -- "${STAGING}"
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_high_confidence_filter.py" \
  --input "${INPUT}" \
  --input-sha256 "${INPUT_SHA256}" \
  --output "${OUTPUT}" \
  --stats "${STATS}" \
  --filter-script "${PROJECT}/scripts/filter_string_v12_high_confidence_physical.sbatch" \
  --protocol "${PROJECT}/protocols/abpairfm_3b_scale_expansion_v2.json" \
  --receipt "${RECEIPT}" \
  --job-id "${SLURM_JOB_ID}" \
  --gpu-name "${GPU_NAME}"
