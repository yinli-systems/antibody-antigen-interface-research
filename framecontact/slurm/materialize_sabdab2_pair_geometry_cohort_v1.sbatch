#!/usr/bin/env bash
#SBATCH --job-name=fc-pair-geom
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pair-geom-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-pair-geom-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/residue-head-source-v1"
archive=/ssd/scxi253/model_deployment/data/pairscale_registered_raw_b6b86f1/sabdab2_ml_zenodo_20083995/splits.tar.gz
cif_root="$base/sabdab2/pair-geometry-cif-cohort-v1"
output_root="$base/sabdab2/pair-geometry-cohort-v1"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3

test ! -e "$cif_root"
test ! -e "$output_root"
mkdir -p "$cif_root"
tar -xzf "$archive" -C "$cif_root" --strip-components=1

cd "$source_root"
export PYTHONPATH="$source_root/src"
"$python_bin" scripts/materialize_sabdab2_pair_geometry.py \
  --structure-cache-root "$base/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "$base/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --cif-root "$cif_root" \
  --metadata "$base/authority/abag_split.csv" \
  --metadata-sha256 01414df16af9d3343994f527e13e81a25bf8da08198858018493a96f173a8cfb \
  --canonical-manifest /ssd/scxi253/model_deployment/data/interaction_first/canonical_manifest.csv \
  --canonical-manifest-sha256 548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63 \
  --output-root "$output_root" \
  --contact-cutoff 4.5 \
  --distance-cap 20.0 \
  --coordinate-quantization-tolerance 0.002 \
  --cdr-flank 2 \
  --antigen-relative-sasa-threshold 0.2

sha256sum \
  "$output_root/PAIR_GEOMETRY_CACHE_MANIFEST.json" \
  "$output_root/PAIR_GEOMETRY_CACHE_RECEIPT.json"
