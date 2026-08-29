#!/usr/bin/env bash
#SBATCH --job-name=fc-audit-fold-v3
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/audit_source_v1"
foldseek_root="${data_root}/foldseek_sharded_4_v1/merged"
split_root="${data_root}/split_matrix_v3"
output_root="${data_root}/structural_audit_v3"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
mkdir -p "${output_root}" "${data_root}/slurm"
exec > >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

"${python_bin}" "${source_root}/scripts/audit_framecontact_structural_split.py" \
  --manifest "${split_root}/antigen_fold.csv" \
  --manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --alignments "${foldseek_root}/foldseek.normalized.tsv" \
  --alignments-sha256 5c49b92e9e0d5d7d07369669a9c463547eb81b3cd1a5babf5df55bb13d108807 \
  --foldseek-receipt "${foldseek_root}/FOLDSEEK_RECEIPT.json" \
  --foldseek-receipt-sha256 99334611536eab76e150c19df4eb32476695989fc6c740ee7e9c6732758d50ad \
  --tm-threshold 0.50 \
  --foldseek-score-rule reciprocal_min \
  --boundary-width 0.02 \
  --boundary-pair-limit 1000 \
  --output "${output_root}/ANTIGEN_FOLD_STRUCTURAL_AUDIT.json"
sha256sum "${output_root}/ANTIGEN_FOLD_STRUCTURAL_AUDIT.json"
