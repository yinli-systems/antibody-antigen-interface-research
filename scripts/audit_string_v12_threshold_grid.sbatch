#!/usr/bin/env bash
#SBATCH -J abpairfm-string-grid
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-grid-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-grid-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly INPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/string_v12/protein.physical.links.full.v12.0.txt.gz
readonly INPUT_SHA256=1d5d16bd8aeb67c9977948a0e56cc27cca2065ad981dab4586330f38d77ff234
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly RECEIPT="${OUTPUT_ROOT}/direct_threshold_grid_v1.receipt.json"
readonly AUDIT_SCRIPT="${PROJECT}/scripts/audit_string_v12_threshold_grid.sbatch"

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
if [[ -e "${RECEIPT}" ]]; then
  echo "Refusing to overwrite STRING threshold evidence" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
readonly STATS=$(mktemp "${OUTPUT_ROOT}/.direct_threshold_grid_v1.XXXXXX.stats")
cleanup_grid() {
  [[ ! -e "${STATS}" ]] || rm -- "${STATS}"
}
trap cleanup_grid EXIT

export LC_ALL=C
gzip -cd "${INPUT}" | awk -v stats="${STATS}" '
  function combined_bin(value) {
    if (value < 600) return "lt600"
    if (value < 650) return "600_649"
    if (value < 700) return "650_699"
    if (value < 750) return "700_749"
    if (value < 800) return "750_799"
    if (value < 900) return "800_899"
    return "ge900"
  }
  function direct_bin(value) {
    if (value == 0) return "zero"
    if (value < 100) return "1_99"
    if (value < 300) return "100_299"
    if (value < 500) return "300_499"
    if (value < 700) return "500_699"
    return "ge700"
  }
  BEGIN {
    combined_names[1]="lt600"; combined_names[2]="600_649"; combined_names[3]="650_699";
    combined_names[4]="700_749"; combined_names[5]="750_799"; combined_names[6]="800_899"; combined_names[7]="ge900";
    direct_names[1]="zero"; direct_names[2]="1_99"; direct_names[3]="100_299";
    direct_names[4]="300_499"; direct_names[5]="500_699"; direct_names[6]="ge700";
  }
  NR == 1 {
    expected = "protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score"
    if ($0 != expected) {
      print "Unexpected STRING header" > "/dev/stderr"
      exit 3
    }
    next
  }
  {
    source_rows++
    if (NF != 10) {
      malformed_rows++
      next
    }
    direct = ($4 > $6 ? $4 : $6)
    bins[combined_bin($10),direct_bin(direct)]++
  }
  END {
    print "source_data_rows=" source_rows > stats
    print "malformed_rows=" malformed_rows + 0 >> stats
    for (i=1; i<=7; i++) {
      for (j=1; j<=6; j++) {
        print "bin_" combined_names[i] "_" direct_names[j] "=" bins[combined_names[i],direct_names[j]] + 0 >> stats
      }
    }
    if (malformed_rows > 0 || source_rows == 0) exit 4
  }
'

"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_threshold_grid.py" \
  --input "${INPUT}" \
  --input-sha256 "${INPUT_SHA256}" \
  --stats "${STATS}" \
  --audit-script "${AUDIT_SCRIPT}" \
  --receipt "${RECEIPT}" \
  --job-id "${SLURM_JOB_ID}" \
  --gpu-name "${GPU_NAME}"
