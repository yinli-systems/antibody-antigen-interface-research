#!/usr/bin/env bash
#SBATCH -J abpairfm-string-cluster
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-cluster-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-cluster-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly RAW_ROOT="${DATA_ROOT}/raw/string_v12"
readonly SOURCE_RECEIPT="${DATA_ROOT}/raw/string_v12.materialization_v1.json"
readonly OUTPUT_ROOT="${DATA_ROOT}/normalized_candidates/string_v12"
readonly DIRECT="${OUTPUT_ROOT}/direct_physical_combined700_v1.tsv.zst"
readonly DIRECT_RECEIPT="${OUTPUT_ROOT}/direct_physical_combined700_v1.receipt.json"
readonly MMSEQS="${DATA_ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs"
readonly ASSETS="${OUTPUT_ROOT}/cluster_assets_v1"
readonly OUTPUT="${OUTPUT_ROOT}/direct_physical_combined700_cluster50_v1.tsv.zst"
readonly RECEIPT="${OUTPUT_ROOT}/direct_physical_combined700_cluster50_v1.receipt.json"
readonly WORK="${OUTPUT_ROOT}/.cluster-work-${SLURM_JOB_ID}"

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING clustering allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
export ABPAIRFM_GPU_COUNT="${GPU_COUNT}"
export ABPAIRFM_GPU_NAME="${GPU_NAME}"

test -x "${MMSEQS}"
test -f "${DIRECT}"
test -f "${DIRECT_RECEIPT}"
test -f "${SOURCE_RECEIPT}"
test ! -e "${ASSETS}"
test ! -e "${OUTPUT}"
test ! -e "${RECEIPT}"
mkdir -p "${WORK}/mmseqs-tmp"

zstd -q -dc "${DIRECT}" \
  | awk 'NR > 1 {print $1; print $2}' \
  | LC_ALL=C sort --parallel=6 -S 20G -u \
  > "${WORK}/endpoint_ids.txt"

"${PYTHON}" "${PROJECT}/scripts/prepare_string_cluster_inputs.py" \
  --endpoint-ids "${WORK}/endpoint_ids.txt" \
  --source-fasta "${RAW_ROOT}/protein.sequences.v12.0.fa.gz" \
  --source-receipt "${SOURCE_RECEIPT}" \
  --output-fasta "${WORK}/endpoint_sequences.fasta" \
  --receipt "${WORK}/endpoint_sequences.receipt.json"

"${MMSEQS}" createdb "${WORK}/endpoint_sequences.fasta" "${WORK}/sequence_db"
"${MMSEQS}" linclust \
  "${WORK}/sequence_db" "${WORK}/cluster_db" "${WORK}/mmseqs-tmp" \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0 --threads 6
"${MMSEQS}" createtsv \
  "${WORK}/sequence_db" "${WORK}/sequence_db" "${WORK}/cluster_db" \
  "${WORK}/clusters.tsv" --threads 6

zstd -q -T6 -19 "${WORK}/endpoint_ids.txt" -o "${WORK}/endpoint_ids.txt.zst"
zstd -q -T6 -19 "${WORK}/endpoint_sequences.fasta" -o "${WORK}/endpoint_sequences.fasta.zst"
zstd -q -T6 -19 "${WORK}/clusters.tsv" -o "${WORK}/clusters.tsv.zst"
chmod -R a-w "${WORK}"
mv "${WORK}" "${ASSETS}"

"${PYTHON}" "${PROJECT}/scripts/collapse_string_cluster_pairs.py" \
  --direct-edges "${DIRECT}" \
  --direct-receipt "${DIRECT_RECEIPT}" \
  --endpoint-receipt "${ASSETS}/endpoint_sequences.receipt.json" \
  --clusters-tsv "${ASSETS}/clusters.tsv" \
  --mmseqs "${MMSEQS}" \
  --asset "${ASSETS}/endpoint_ids.txt.zst" \
  --asset "${ASSETS}/endpoint_sequences.fasta.zst" \
  --asset "${ASSETS}/clusters.tsv.zst" \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}" \
  --threads 6
