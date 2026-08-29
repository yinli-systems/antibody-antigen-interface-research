#!/usr/bin/env bash
#SBATCH --job-name=fc-double-cold-v4
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-double-cold-v4-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-double-cold-v4-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/double-cold-source-v5"
foldseek_root="$base/foldseek_sharded_4_v1/merged"
output_root="$base/split_matrix_double_cold_v4"
python_bin=/ssd/scxi253/model_deployment/envs/epiformer/bin/python

if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite double-cold split: $output_root" >&2
  exit 2
fi
mkdir -p "$base/slurm-logs"
cd "$source_root"
export PYTHONPATH="$source_root/src"

"$python_bin" scripts/build_sabdab2_split_matrix.py \
  --metadata "$base/authority/abag_split.csv" \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --prepared "$base/authority/instances.parquet" \
  --prepared-sha256 a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335 \
  --foldseek-alignments "$foldseek_root/foldseek.normalized.tsv" \
  --foldseek-alignments-sha256 5c49b92e9e0d5d7d07369669a9c463547eb81b3cd1a5babf5df55bb13d108807 \
  --foldseek-receipt "$foldseek_root/FOLDSEEK_RECEIPT.json" \
  --foldseek-receipt-sha256 99334611536eab76e150c19df4eb32476695989fc6c740ee7e9c6732758d50ad \
  --train-cutoff 2022-12-31 \
  --validation-cutoff 2024-06-30 \
  --tm-threshold 0.50 \
  --foldseek-score-rule reciprocal_min \
  --metadata-cluster-policy double_cold \
  --assignment-seed 20260730 \
  --output-root "$output_root"

sha256sum "$output_root"/*.csv "$output_root/SPLIT_MATRIX_RECEIPT.json" \
  > "$output_root/SHA256SUMS.txt"

"$python_bin" scripts/audit_sabdab2_split_shortcuts.py \
  --split-manifest "$output_root/antigen_fold.csv" \
  --split-manifest-sha256 "$(sha256sum "$output_root/antigen_fold.csv" | cut -d' ' -f1)" \
  --output "$output_root/SHORTCUT_AUDIT.json"

sha256sum "$output_root/SHORTCUT_AUDIT.json" >> "$output_root/SHA256SUMS.txt"
