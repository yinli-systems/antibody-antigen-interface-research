#!/usr/bin/env bash
#SBATCH --job-name=fc-foldseek-grid
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/split_source_v2"
foldseek_root="${data_root}/foldseek_sharded_4_v1/merged"
output_root="${data_root}/split_audits_v1"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
mkdir -p "${output_root}" "${data_root}/slurm"
exec > >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

"${python_bin}" "${source_root}/scripts/audit_foldseek_component_grid.py" \
  --metadata "${data_root}/authority/abag_split.csv" \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --prepared "${data_root}/authority/instances.parquet" \
  --prepared-sha256 a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335 \
  --foldseek-alignments "${foldseek_root}/foldseek.normalized.tsv" \
  --foldseek-alignments-sha256 5c49b92e9e0d5d7d07369669a9c463547eb81b3cd1a5babf5df55bb13d108807 \
  --foldseek-receipt "${foldseek_root}/FOLDSEEK_RECEIPT.json" \
  --foldseek-receipt-sha256 99334611536eab76e150c19df4eb32476695989fc6c740ee7e9c6732758d50ad \
  --assignment-seed 20260730 \
  --output "${output_root}/FOLDSEEK_COMPONENT_GRID.json"
sha256sum "${output_root}/FOLDSEEK_COMPONENT_GRID.json"
