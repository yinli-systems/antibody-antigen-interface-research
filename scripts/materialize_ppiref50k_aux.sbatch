#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref-aux
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-aux-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-aux-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/ppiref50k_aux_20260816
readonly COMMIT=a2a5b6d01f958acac39418bda1a5a9e3b4d71b53
readonly SPLIT_URL="https://raw.githubusercontent.com/anton-bushuiev/PPIRef/${COMMIT}/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json"
readonly SEQRES_URL=https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz
readonly SPLIT=${OUTPUT_ROOT}/ppiref_6A_filtered_clustered_04.json
readonly SEQRES=${OUTPUT_ROOT}/pdb_seqres_20260816.txt.gz
readonly RECEIPT=${OUTPUT_ROOT}/materialization_receipt.json

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden materialization allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
for path in "${SPLIT}" "${SEQRES}" "${RECEIPT}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite PPIRef auxiliary evidence" >&2; exit 2; }
done
readonly SPLIT_STAGE=$(mktemp "${OUTPUT_ROOT}/.split.XXXXXX")
readonly SEQRES_STAGE=$(mktemp "${OUTPUT_ROOT}/.seqres.XXXXXX")
readonly SPLIT_HEADERS=$(mktemp "${OUTPUT_ROOT}/.split-headers.XXXXXX")
readonly SEQRES_HEADERS=$(mktemp "${OUTPUT_ROOT}/.seqres-headers.XXXXXX")
cleanup_aux() {
  rm -f -- "${SPLIT_STAGE}" "${SEQRES_STAGE}" "${SPLIT_HEADERS}" "${SEQRES_HEADERS}"
}
trap cleanup_aux EXIT

curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
  --dump-header "${SPLIT_HEADERS}" --output "${SPLIT_STAGE}" "${SPLIT_URL}"
printf '%s  %s\n' a77239af16efd8c4a83353405ba2e6670720270955d9831febacbcdbad92ce57 "${SPLIT_STAGE}" \
  | sha256sum --check --strict
curl --fail --location --retry 5 --retry-all-errors --connect-timeout 20 \
  --dump-header "${SEQRES_HEADERS}" --output "${SEQRES_STAGE}" "${SEQRES_URL}"
gzip --test "${SEQRES_STAGE}"

chmod 0444 "${SPLIT_STAGE}" "${SEQRES_STAGE}"
ln "${SPLIT_STAGE}" "${SPLIT}"
ln "${SEQRES_STAGE}" "${SEQRES}"
"${PYTHON}" "${PROJECT}/scripts/finalize_ppiref50k_aux_materialization.py" \
  --ppiref-split "${SPLIT}" \
  --ppiref-headers "${SPLIT_HEADERS}" \
  --seqres "${SEQRES}" \
  --seqres-headers "${SEQRES_HEADERS}" \
  --receipt "${RECEIPT}" \
  --job-id "${SLURM_JOB_ID}" \
  --gpu-name "${GPU_NAME}"
