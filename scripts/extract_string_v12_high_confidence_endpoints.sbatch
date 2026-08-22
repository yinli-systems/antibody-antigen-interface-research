#!/usr/bin/env bash
#SBATCH -J abpairfm-string-high700-ids
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-ids-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-ids-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/string_v12
readonly EDGES=${ROOT}/physical_combined700_all_channels_v2.tsv.zst
readonly EDGE_RECEIPT=${ROOT}/physical_combined700_all_channels_v2.receipt.json
readonly OUTPUT=${ROOT}/physical_combined700_all_channels_endpoint_ids_v1.txt.zst
readonly RECEIPT=${ROOT}/physical_combined700_all_channels_endpoint_ids_v1.receipt.json
readonly WORK=$(mktemp -d "${ROOT}/.high700-endpoints-${SLURM_JOB_ID}.XXXXXX")
readonly STAGING=${WORK}/endpoint_ids.txt.zst
readonly COUNT=${WORK}/endpoint_count.txt
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING endpoint allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" || -e "${RECEIPT}" ]]; then
  echo "Refusing to overwrite STRING endpoint evidence" >&2
  exit 2
fi
cleanup_endpoints() {
  rm -rf -- "${WORK}"
}
trap cleanup_endpoints EXIT

export LC_ALL=C
zstd -q -dc "${EDGES}" | \
  awk 'NR == 1 { if ($1 != "protein1" || $2 != "protein2" || NF != 10) exit 3; next } { if (NF != 10) exit 4; print $1; print $2 }' | \
  sort --parallel=6 -S 40G -T "${WORK}" -u | \
  awk -v count="${COUNT}" '{ rows++; print } END { print rows + 0 > count; if (rows == 0) exit 5 }' | \
  zstd -T4 -3 --check --force -o "${STAGING}"

zstd --test "${STAGING}"
chmod 0444 "${STAGING}"
ln "${STAGING}" "${OUTPUT}"
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_high_confidence_endpoints.py" \
  --edges "${EDGES}" \
  --edge-receipt "${EDGE_RECEIPT}" \
  --endpoint-ids "${OUTPUT}" \
  --endpoint-count "${COUNT}" \
  --receipt "${RECEIPT}"
