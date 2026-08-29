#!/usr/bin/env bash
#SBATCH -J abpairfm-string-high700-dev30
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=08:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-dev30-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-high700-dev30-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly ROOT=${DATA_ROOT}/normalized_candidates/string_v12
readonly ASSETS=${ROOT}/cluster_assets_high700_v1
readonly ENDPOINT_FASTA=${ASSETS}/endpoint_sequences.fasta.zst
readonly CLUSTER_RECEIPT=${ASSETS}/cluster_assets.receipt.json
readonly PANEL=${DATA_ROOT}/development_anchor
readonly MMSEQS=${DATA_ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly OUTPUT=${ROOT}/strict_development30_high700_v1
readonly WORK=$(mktemp -d "${ROOT}/.strict-development30-${SLURM_JOB_ID}.XXXXXX")
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden strict decontamination allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test ! -e "${OUTPUT}"
for path in "${PYTHON}" "${MMSEQS}" "${ENDPOINT_FASTA}" "${CLUSTER_RECEIPT}" \
  "${PANEL}/development_sequences_v1.fasta" "${PANEL}/development_sequences_v1.receipt.json"; do
  test -e "${path}"
done
cleanup_strict_search() {
  [[ ! -d "${WORK}" ]] || rm -r -- "${WORK}"
}
trap cleanup_strict_search EXIT
mkdir -p "${WORK}/mmseqs-tmp"

zstd -q -dc "${ENDPOINT_FASTA}" > "${WORK}/endpoint_sequences.fasta"
"${MMSEQS}" createdb "${WORK}/endpoint_sequences.fasta" "${WORK}/endpoint_db"
"${MMSEQS}" createdb "${PANEL}/development_sequences_v1.fasta" "${WORK}/development_db"
"${MMSEQS}" search \
  "${WORK}/development_db" "${WORK}/endpoint_db" "${WORK}/hits_db" "${WORK}/mmseqs-tmp" \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 --alignment-mode 3 \
  --max-seqs 1000000 -s 7.0 --threads 6
"${MMSEQS}" convertalis \
  "${WORK}/development_db" "${WORK}/endpoint_db" "${WORK}/hits_db" \
  "${WORK}/development_hits30.tsv" \
  --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits
zstd -q -T6 -3 --check "${WORK}/development_hits30.tsv" \
  -o "${WORK}/development_hits30.tsv.zst"
"${PYTHON}" "${PROJECT}/scripts/finalize_string_v12_strict_development_hits.py" \
  --endpoint-fasta "${ENDPOINT_FASTA}" \
  --cluster-receipt "${CLUSTER_RECEIPT}" \
  --development-fasta "${PANEL}/development_sequences_v1.fasta" \
  --development-receipt "${PANEL}/development_sequences_v1.receipt.json" \
  --hits "${WORK}/development_hits30.tsv.zst" \
  --output-name development_hits30.tsv.zst \
  --mmseqs-version "$("${MMSEQS}" version)" \
  --receipt "${WORK}/strict_development_hits.receipt.json"
rm -f -- "${WORK}/endpoint_sequences.fasta" "${WORK}/development_hits30.tsv"
rm -rf -- "${WORK}/endpoint_db"* "${WORK}/development_db"* \
  "${WORK}/hits_db"* "${WORK}/mmseqs-tmp"
chmod -R a-w "${WORK}"
mv "${WORK}" "${OUTPUT}"
