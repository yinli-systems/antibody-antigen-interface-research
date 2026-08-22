#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref50k
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref50k-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref50k-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly RAW=/ssd/scxi253/model_deployment/data/pairscale_registered_raw_b6b86f1/ppiref_zenodo_14845086
readonly AUX=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/ppiref50k_aux_20260816
readonly OUTPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/ppiref50k_2024_01
readonly ARCHIVE_SHA256=dece33d55f1098c699fdddeac3e1cb38fa23c65f9eac36fec7d339190ddc8cd3

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden preprocessing allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"
"${PYTHON}" "${PROJECT}/scripts/build_ppiref50k_interfaces_v2.py" \
  --archive "${RAW}/ppi_6A.zip" \
  --archive-sha256 "${ARCHIVE_SHA256}" \
  --split "${AUX}/ppiref_6A_filtered_clustered_04.json" \
  --aux-receipt "${AUX}/materialization_receipt.json" \
  --coverage-receipt "${OUTPUT}/archive_coverage_audit_v2.json" \
  --seqres "${AUX}/pdb_seqres_20260816.txt.gz" \
  --manifest-output "${OUTPUT}/interfaces_v2.jsonl.zst" \
  --fasta-output "${OUTPUT}/full_sequences_v2.fasta" \
  --receipt "${OUTPUT}/normalization_receipt_v2.json" \
  --minimum-contact-pairs 8 \
  --contact-radius 6.0 \
  --threads 6
