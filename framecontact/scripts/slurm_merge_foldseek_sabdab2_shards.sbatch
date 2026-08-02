#!/usr/bin/env bash
#SBATCH --job-name=fc-foldseek-merge
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
sharded_root="${data_root}/foldseek_sharded_4_v1"
plan="${sharded_root}/query-plan/FOLDSEEK_SHARD_PLAN.json"
export_manifest="${data_root}/antigen_structures_full_v1/ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv"
merged_root="${sharded_root}/merged"
log_root="${data_root}/slurm"
mkdir -p "${log_root}" "${merged_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

foldseek="/ssd/scxi253/model_deployment/tools/foldseek-official-avx2-20260730/foldseek/bin/foldseek"
python3 "${source_root}/scripts/merge_foldseek_shards.py" \
  --shard-output-root "${sharded_root}/shard-outputs" \
  --plan "${plan}" \
  --plan-sha256 "$(sha256sum "${plan}" | awk '{print $1}')" \
  --export-manifest "${export_manifest}" \
  --export-manifest-sha256 "$(sha256sum "${export_manifest}" | awk '{print $1}')" \
  --raw-output "${merged_root}/foldseek.raw.tsv" \
  --normalized-output "${merged_root}/foldseek.normalized.tsv" \
  --receipt "${merged_root}/FOLDSEEK_RECEIPT.json" \
  --foldseek-version "$("${foldseek}" version)"
sha256sum "${merged_root}/foldseek.raw.tsv" \
  "${merged_root}/foldseek.normalized.tsv" \
  "${merged_root}/FOLDSEEK_RECEIPT.json" > "${merged_root}/SHA256SUMS.txt"
cat "${merged_root}/FOLDSEEK_RECEIPT.json"
