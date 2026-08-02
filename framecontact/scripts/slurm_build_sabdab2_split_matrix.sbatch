#!/usr/bin/env bash
#SBATCH --job-name=fc-split-matrix
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
foldseek_root="${data_root}/foldseek_sharded_4_v1/merged"
output_root="${data_root}/split_matrix_v1"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
log_root="${data_root}/slurm"
mkdir -p "${log_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

normalized="${foldseek_root}/foldseek.normalized.tsv"
receipt="${foldseek_root}/FOLDSEEK_RECEIPT.json"
"${python_bin}" "${source_root}/scripts/build_sabdab2_split_matrix.py" \
  --metadata "${data_root}/authority/abag_split.csv" \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --prepared "${data_root}/authority/instances.parquet" \
  --prepared-sha256 a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335 \
  --foldseek-alignments "${normalized}" \
  --foldseek-alignments-sha256 "$(sha256sum "${normalized}" | awk '{print $1}')" \
  --foldseek-receipt "${receipt}" \
  --foldseek-receipt-sha256 "$(sha256sum "${receipt}" | awk '{print $1}')" \
  --train-cutoff 2022-12-31 \
  --validation-cutoff 2024-06-30 \
  --tm-threshold 0.50 \
  --assignment-seed 20260730 \
  --output-root "${output_root}"
sha256sum "${output_root}"/*.csv "${output_root}/SPLIT_MATRIX_RECEIPT.json" \
  > "${output_root}/SHA256SUMS.txt"
