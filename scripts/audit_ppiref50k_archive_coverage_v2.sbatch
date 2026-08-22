#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-audit-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-audit-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly ARCHIVE=/ssd/scxi253/model_deployment/data/pairscale_registered_raw_b6b86f1/ppiref_zenodo_14845086/ppi_6A.zip
readonly SPLIT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/ppiref50k_aux_20260816/ppiref_6A_filtered_clustered_04.json
readonly OUTPUT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/ppiref50k_2024_01/archive_coverage_audit_v2.json

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test ! -e "${OUTPUT}"
"${PYTHON}" "${PROJECT}/scripts/audit_ppiref50k_archive_coverage_v2.py" \
  --archive "${ARCHIVE}" \
  --archive-sha256 dece33d55f1098c699fdddeac3e1cb38fa23c65f9eac36fec7d339190ddc8cd3 \
  --split "${SPLIT}" \
  --split-sha256 a77239af16efd8c4a83353405ba2e6670720270955d9831febacbcdbad92ce57 \
  --output "${OUTPUT}"
