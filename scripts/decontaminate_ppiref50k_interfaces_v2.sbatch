#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref-decontam
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-decontam-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-decontam-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
readonly SOURCE=${ROOT}/normalized_candidates/ppiref50k_2024_01
readonly PANEL=${ROOT}/development_anchor
readonly MMSEQS=${ROOT}/tools/mmseqs2-f71d0a6b57c1/bin/mmseqs
readonly WORK=${SOURCE}/.decontam-work-${SLURM_JOB_ID}

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
for path in \
  "${SOURCE}/interfaces_v2.jsonl.zst" "${SOURCE}/full_sequences_v2.fasta" \
  "${SOURCE}/normalization_receipt_v2.json" \
  "${PANEL}/development_sequences_v1.fasta" "${PANEL}/development_sequences_v1.receipt.json" \
  "${MMSEQS}"; do
  test -e "${path}"
done
for path in \
  "${SOURCE}/interfaces_devclean_v2.jsonl.zst" "${SOURCE}/full_sequences_devclean_v2.fasta" \
  "${SOURCE}/development_homology_hits_v2.tsv" "${SOURCE}/development_excluded_interfaces_v2.txt" \
  "${SOURCE}/development_decontamination_receipt_v2.json"; do
  test ! -e "${path}"
done
mkdir -p "${WORK}/tmp"
"${MMSEQS}" createdb "${PANEL}/development_sequences_v1.fasta" "${WORK}/development_db"
"${MMSEQS}" createdb "${SOURCE}/full_sequences_v2.fasta" "${WORK}/ppiref_db"
"${MMSEQS}" search "${WORK}/development_db" "${WORK}/ppiref_db" "${WORK}/hits_db" "${WORK}/tmp" \
  --min-seq-id 0.5 -c 0.8 --cov-mode 0 --alignment-mode 3 --max-seqs 1000000 -s 7.0 --threads 6
"${MMSEQS}" convertalis "${WORK}/development_db" "${WORK}/ppiref_db" "${WORK}/hits_db" "${WORK}/hits.tsv" \
  --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits

"${PYTHON}" "${PROJECT}/scripts/decontaminate_ppiref50k_interfaces_v2.py" \
  --manifest "${SOURCE}/interfaces_v2.jsonl.zst" \
  --fasta "${SOURCE}/full_sequences_v2.fasta" \
  --normalization-receipt "${SOURCE}/normalization_receipt_v2.json" \
  --development-fasta "${PANEL}/development_sequences_v1.fasta" \
  --development-receipt "${PANEL}/development_sequences_v1.receipt.json" \
  --hits "${WORK}/hits.tsv" \
  --hits-output "${SOURCE}/development_homology_hits_v2.tsv" \
  --excluded-output "${SOURCE}/development_excluded_interfaces_v2.txt" \
  --manifest-output "${SOURCE}/interfaces_devclean_v2.jsonl.zst" \
  --fasta-output "${SOURCE}/full_sequences_devclean_v2.fasta" \
  --receipt "${SOURCE}/development_decontamination_receipt_v2.json" \
  --mmseqs-version "$("${MMSEQS}" version)" \
  --threads 6
