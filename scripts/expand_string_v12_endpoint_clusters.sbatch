#!/usr/bin/env bash
#SBATCH -J abpairfm-string-expand-clusters
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-expand-clusters-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-string-expand-clusters-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly DATA_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly RAW_ROOT=${DATA_ROOT}/raw/string_v12
readonly SOURCE_RECEIPT=${DATA_ROOT}/raw/string_v12.materialization_v1.json
readonly ROOT=${DATA_ROOT}/normalized_candidates/string_v12
readonly DIRECT=${ROOT}/direct_physical_combined700_v1.tsv.zst
readonly DIRECT_RECEIPT=${ROOT}/direct_physical_combined700_v1.receipt.json
readonly SUPPLEMENT=${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.tsv.zst
readonly SUPPLEMENT_RECEIPT=${ROOT}/direct_physical_combined600_699_supplemental_universe_v1.receipt.json
readonly PANEL=${DATA_ROOT}/development_anchor
readonly MMSEQS=${DATA_ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly ASSETS=${ROOT}/cluster_assets_expanded_v1
readonly CLUSTERED=${ROOT}/direct_physical_combined700_cluster50_expanded_v1.tsv.zst
readonly CLUSTERED_RECEIPT=${ROOT}/direct_physical_combined700_cluster50_expanded_v1.receipt.json
readonly DEVCLEAN=${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.tsv.zst
readonly DEVCLEAN_RECEIPT=${ROOT}/direct_physical_combined700_cluster50_expanded_devclean_v1.receipt.json
readonly HITS_OUTPUT=${ROOT}/development_homology_hits_expanded_v1.tsv
readonly EXCLUDED_OUTPUT=${ROOT}/development_excluded_clusters_expanded_v1.txt
readonly WORK=${ROOT}/.expanded-cluster-work-${SLURM_JOB_ID}

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden STRING clustering allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
export ABPAIRFM_GPU_COUNT=${GPU_COUNT}
export ABPAIRFM_GPU_NAME=${GPU_NAME}

for path in "${DIRECT}" "${DIRECT_RECEIPT}" "${SUPPLEMENT}" "${SUPPLEMENT_RECEIPT}" "${SOURCE_RECEIPT}" "${PANEL}/development_sequences_v1.fasta" "${PANEL}/development_sequences_v1.receipt.json"; do
  test -f "${path}"
done
test -x "${MMSEQS}"
for path in "${ASSETS}" "${CLUSTERED}" "${CLUSTERED_RECEIPT}" "${DEVCLEAN}" "${DEVCLEAN_RECEIPT}" "${HITS_OUTPUT}" "${EXCLUDED_OUTPUT}"; do
  test ! -e "${path}"
done
mkdir -p "${WORK}/mmseqs-tmp"

"${PYTHON}" "${PROJECT}/scripts/build_string_v12_expanded_endpoint_union.py" \
  --direct "${DIRECT}" \
  --direct-receipt "${DIRECT_RECEIPT}" \
  --supplemental "${SUPPLEMENT}" \
  --supplemental-receipt "${SUPPLEMENT_RECEIPT}" \
  --output "${WORK}/endpoint_ids.txt" \
  --receipt "${WORK}/endpoint_union.receipt.json"

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
  --asset "${ASSETS}/endpoint_union.receipt.json" \
  --output "${CLUSTERED}" \
  --receipt "${CLUSTERED_RECEIPT}" \
  --threads 6

"${PYTHON}" "${PROJECT}/scripts/decontaminate_string_cluster_pairs.py" \
  --clustered-edges "${CLUSTERED}" \
  --cluster-receipt "${CLUSTERED_RECEIPT}" \
  --clusters-tsv "${ASSETS}/clusters.tsv" \
  --development-fasta "${PANEL}/development_sequences_v1.fasta" \
  --development-receipt "${PANEL}/development_sequences_v1.receipt.json" \
  --hits "${ASSETS}/development_hits.tsv" \
  --hits-output "${HITS_OUTPUT}" \
  --excluded-clusters-output "${EXCLUDED_OUTPUT}" \
  --output "${DEVCLEAN}" \
  --receipt "${DEVCLEAN_RECEIPT}" \
  --mmseqs-version "$("${MMSEQS}" version)" \
  --threads 6
