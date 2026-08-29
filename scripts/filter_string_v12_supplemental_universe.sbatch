#!/usr/bin/env bash
#SBATCH -J abpairfm-string-supplement
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-supplement-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-supplement-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly INPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/string_v12/protein.physical.links.full.v12.0.txt.gz
readonly INPUT_SHA256=1d5d16bd8aeb67c9977948a0e56cc27cca2065ad981dab4586330f38d77ff234
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly GRID=${ROOT}/direct_threshold_grid_v1.receipt.json
readonly OUTPUT=${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.tsv.zst
readonly RECEIPT=${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.receipt.json
readonly FILTER_SCRIPT=${PROJECT}/scripts/filter_string_v12_supplemental_universe.sbatch

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -f "${GRID}"
for path in "${OUTPUT}" "${RECEIPT}"; do
  test ! -e "${path}"
done
readonly STAGING=$(mktemp "${ROOT}/.supplemental-universe.XXXXXX.tsv.zst")
readonly STATS=$(mktemp "${ROOT}/.supplemental-universe.XXXXXX.stats")
cleanup_supplement() {
  rm -f -- "${STAGING}" "${STATS}"
}
trap cleanup_supplement EXIT

export LC_ALL=C
gzip -cd "${INPUT}" | awk -v stats="${STATS}" '
  NR == 1 {
    expected = "protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score"
    if ($0 != expected) { print "Unexpected STRING header" > "/dev/stderr"; exit 3 }
    print $0
    next
  }
  {
    source_rows++
    if (NF != 10) { malformed_rows++; next }
    direct = ($4 > $6 ? $4 : $6)
    if ($10 >= 600 && $10 < 700 && direct >= 1) {
      candidate_rows++
      print $0
    }
  }
  END {
    print "candidate_rows=" candidate_rows + 0 > stats
    print "malformed_rows=" malformed_rows + 0 >> stats
    print "source_data_rows=" source_rows + 0 >> stats
    if (malformed_rows > 0 || candidate_rows == 0) exit 4
  }
' | zstd -T4 -3 --check --force -o "${STAGING}"
zstd --test "${STAGING}"
chmod 0444 "${STAGING}"
ln "${STAGING}" "${OUTPUT}"
rm -- "${STAGING}"
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_supplemental_universe.py" \
  --input "${INPUT}" --input-sha256 "${INPUT_SHA256}" \
  --output "${OUTPUT}" --stats "${STATS}" \
  --threshold-grid-receipt "${GRID}" --filter-script "${FILTER_SCRIPT}" \
  --receipt "${RECEIPT}" --job-id "${SLURM_JOB_ID}" --gpu-name "${GPU_NAME}"
