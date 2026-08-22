#!/usr/bin/env bash
#SBATCH -J abpairfm-string-high700-cluster
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-cluster-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-cluster-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly RAW_ROOT=${DATA_ROOT}/raw/string_v12
readonly SOURCE_RECEIPT=${DATA_ROOT}/raw/string_v12.materialization_v1.json
readonly ROOT=${DATA_ROOT}/normalized_candidates/string_v12
readonly ENDPOINT_IDS=${ROOT}/physical_combined700_all_channels_endpoint_ids_v1.txt.zst
readonly ENDPOINT_RECEIPT=${ROOT}/physical_combined700_all_channels_endpoint_ids_v1.receipt.json
readonly PANEL=${DATA_ROOT}/development_anchor
readonly MMSEQS=${DATA_ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly ASSETS=${ROOT}/cluster_assets_high700_v1
readonly WORK=$(mktemp -d "${ROOT}/.high700-cluster-${SLURM_JOB_ID}.XXXXXX")
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING clustering allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test ! -e "${ASSETS}"
for path in "${ENDPOINT_IDS}" "${ENDPOINT_RECEIPT}" "${SOURCE_RECEIPT}" "${PANEL}/development_sequences_v1.fasta" "${PANEL}/development_sequences_v1.receipt.json"; do
  test -f "${path}"
done
test -x "${MMSEQS}"
mkdir -p "${WORK}/mmseqs-tmp"

zstd -q -dc "${ENDPOINT_IDS}" > "${WORK}/endpoint_ids.txt"
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

"${MMSEQS}" createdb "${PANEL}/development_sequences_v1.fasta" "${WORK}/development_db"
"${MMSEQS}" search \
  "${WORK}/development_db" "${WORK}/sequence_db" "${WORK}/development_hits_db" "${WORK}/mmseqs-tmp" \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 1000000 -s 7.0 --threads 6
"${MMSEQS}" convertalis \
  "${WORK}/development_db" "${WORK}/sequence_db" "${WORK}/development_hits_db" \
  "${WORK}/development_hits.tsv" \
  --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits

awk 'NF != 2 { exit 3 } { rows++ } END { print rows + 0 }' "${WORK}/clusters.tsv" > "${WORK}/membership_count.txt"
zstd -q -T6 -3 "${WORK}/endpoint_ids.txt" -o "${WORK}/endpoint_ids.txt.zst"
zstd -q -T6 -3 "${WORK}/endpoint_sequences.fasta" -o "${WORK}/endpoint_sequences.fasta.zst"
zstd -q -T6 -3 "${WORK}/clusters.tsv" -o "${WORK}/clusters.tsv.zst"
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_high_confidence_clusters.py" \
  --endpoint-ids "${ENDPOINT_IDS}" \
  --endpoint-receipt "${ENDPOINT_RECEIPT}" \
  --endpoint-fasta-receipt "${WORK}/endpoint_sequences.receipt.json" \
  --clusters "${WORK}/clusters.tsv.zst" \
  --membership-count "${WORK}/membership_count.txt" \
  --development-hits "${WORK}/development_hits.tsv" \
  --mmseqs-version "$("${MMSEQS}" version)" \
  --asset "${WORK}/endpoint_ids.txt.zst" \
  --asset "${WORK}/endpoint_sequences.fasta.zst" \
  --asset "${WORK}/clusters.tsv.zst" \
  --asset "${WORK}/development_hits.tsv" \
  --receipt "${WORK}/cluster_assets.receipt.json"

chmod -R a-w "${WORK}"
mv "${WORK}" "${ASSETS}"
