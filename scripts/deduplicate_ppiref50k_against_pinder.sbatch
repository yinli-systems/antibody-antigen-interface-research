#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref-pinder-dedup
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-pinder-dedup-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-pinder-dedup-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates
readonly PINDER=${ROOT}/pinder_2024_02
readonly PPIREF=${ROOT}/ppiref50k_2024_01

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi

"${PYTHON}" "${PROJECT}/scripts/deduplicate_ppiref50k_against_pinder.py" \
  --pinder-systems "${PINDER}/train_systems_devclean_v1.tsv.gz" \
  --pinder-systems-receipt "${PINDER}/train_systems_devclean_v1.receipt.json" \
  --pinder-sequences "${PINDER}/pinder_holo_sequences_v3.tsv.gz" \
  --pinder-sequences-receipt "${PINDER}/pinder_holo_sequences_v3.receipt.json" \
  --ppiref-manifest "${PPIREF}/interfaces_devclean_v2.jsonl.zst" \
  --ppiref-fasta "${PPIREF}/full_sequences_devclean_v2.fasta" \
  --ppiref-receipt "${PPIREF}/development_decontamination_receipt_v2.json" \
  --manifest-output "${PPIREF}/interfaces_devclean_pindernovel_v1.jsonl.zst" \
  --fasta-output "${PPIREF}/full_sequences_devclean_pindernovel_v1.fasta" \
  --exclusions-output "${PPIREF}/pinder_priority_exclusions_v1.tsv" \
  --receipt "${PPIREF}/pinder_priority_deduplication_receipt_v1.json" \
  --threads 6
