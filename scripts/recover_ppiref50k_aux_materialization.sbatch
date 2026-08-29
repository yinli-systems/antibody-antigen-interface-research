#!/usr/bin/env bash
#SBATCH -J abpairfm-ppiref-aux-recovery
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-aux-recovery-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-ppiref-aux-recovery-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly OUTPUT_ROOT=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/raw/ppiref50k_aux_20260816
readonly INCOMING=${OUTPUT_ROOT}/incoming_local_transfer_v1
readonly SPLIT=${OUTPUT_ROOT}/ppiref_6A_filtered_clustered_04.json
readonly SEQRES=${OUTPUT_ROOT}/pdb_seqres_20260816.txt.gz
readonly RECEIPT=${OUTPUT_ROOT}/materialization_receipt.json
readonly SPLIT_SHA256=a77239af16efd8c4a83353405ba2e6670720270955d9831febacbcdbad92ce57
readonly SEQRES_SHA256=78f793b07640925c6e65a2534c00e1d61dc12101ca711d5a604d5306ad6e87ce

readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden recovery allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
for path in "${SPLIT}" "${SEQRES}" "${RECEIPT}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite PPIRef auxiliary evidence" >&2; exit 2; }
done
printf '%s  %s\n' "${SPLIT_SHA256}" "${INCOMING}/ppiref_6A_filtered_clustered_04.json" \
  "${SEQRES_SHA256}" "${INCOMING}/pdb_seqres_20260816.txt.gz" \
  | sha256sum --check --strict
gzip --test "${INCOMING}/pdb_seqres_20260816.txt.gz"
chmod 0444 "${INCOMING}/ppiref_6A_filtered_clustered_04.json" \
  "${INCOMING}/pdb_seqres_20260816.txt.gz" \
  "${INCOMING}/split.headers" "${INCOMING}/seqres.headers"
ln "${INCOMING}/ppiref_6A_filtered_clustered_04.json" "${SPLIT}"
ln "${INCOMING}/pdb_seqres_20260816.txt.gz" "${SEQRES}"

"${PYTHON}" "${PROJECT}/scripts/finalize_ppiref50k_aux_materialization.py" \
  --ppiref-split "${SPLIT}" \
  --ppiref-headers "${INCOMING}/split.headers" \
  --seqres "${SEQRES}" \
  --seqres-headers "${INCOMING}/seqres.headers" \
  --receipt "${RECEIPT}" \
  --job-id "${SLURM_JOB_ID}" \
  --gpu-name "${GPU_NAME}" \
  --transport-mode authenticated_local_https_scp_recovery \
  --failed-job-id 1458745
