#!/usr/bin/env bash
#SBATCH --job-name=fc-pplm-real-smoke
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pplm-real-smoke-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pplm-real-smoke-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/frontier-source-v4"
source_manifest="$base/source-manifests/frontier-source-v4.json"
source_manifest_sha256="$1"
checkpoint_sha256="$2"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3
pplm_source=/ssd/scxi253/model_deployment/external/PPLM-1267db4592703095b889cc22da6cf1796ce038b5
checkpoint=/ssd/scxi253/model_deployment/models/pplm/revision-1267db4592703095b889cc22da6cf1796ce038b5/pplm_t33_650M.pt
fixture_root="$source_root/protocols/fixtures/pplm_smoke"
fixture_receipt="$fixture_root/pdb_00008c3k_B_+_receipt.json"
output_root="$base/smoke/pplm-real-checkpoint-v1-${SLURM_JOB_ID}"

cd "$source_root"
export PYTHONPATH="$source_root/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$python_bin" scripts/freeze_training_source_tree.py verify \
  --source-root "$source_root" \
  --manifest "$source_manifest" \
  --manifest-sha256 "$source_manifest_sha256"

mkdir -p "$output_root"
"$python_bin" scripts/extract_pplm_paired_features.py \
  --pplm-source-root "$pplm_source" \
  --checkpoint "$checkpoint" \
  --checkpoint-sha256 "$checkpoint_sha256" \
  --antibody-fasta "$fixture_root/pdb_00008c3k_B_+_antibody.fasta" \
  --antigen-fasta "$fixture_root/pdb_00008c3k_B_+_antigen.fasta" \
  --fixture-receipt "$fixture_receipt" \
  --fixture-receipt-sha256 bb0d97acd6a525d75912de6df33bf721973a693bf76241f7a075cead70acdcf2 \
  --output "$output_root/pair.npz" \
  --receipt "$output_root/PAIR_RECEIPT.json" \
  --device cuda

"$python_bin" -c 'from interaction_first.paired_sequence_features import load_pplm_pair_feature_artifact; import sys; result=load_pplm_pair_feature_artifact(sys.argv[1], sys.argv[2]); print({key: value.shape for key, value in result.items() if hasattr(value, "shape")})' "$output_root/pair.npz" "$output_root/PAIR_RECEIPT.json"
sha256sum "$output_root/pair.npz" "$output_root/PAIR_RECEIPT.json"
