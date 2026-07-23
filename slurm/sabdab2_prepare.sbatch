#!/bin/bash
#SBATCH --job-name=sabdab2-prepare
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

R=/cluster/project/model_deployment/data/sabdab2-baseline-outsourcing
F=$R/data/nfs/yangwuyu/baseline-eval/frozen/sabdab2
D=$R/data/nfs/yangwuyu/sabdab2_bipartite_contacts
P=$R/data/nfs/yangwuyu/baseline-eval/datasets/pdb

mkdir -p "$F" "$D/canonical" "$D/labels" "$P"

test -s "$F/MANIFEST.json" || tar -xzf "$R/sabdab2-frozen-20260721.tar.gz" -C "$F"
test -s "$D/canonical/manifest.csv" || tar -xzf "$R/sabdab2-canonical-20260721.tar.gz" -C "$D/canonical"
test -s "$D/labels/index.csv" || tar -xzf "$R/sabdab2-labels-20260721.tar.gz" -C "$D/labels"
if ! find "$P" -maxdepth 1 -type f \( -name '*.pdb' -o -name '*.cif' \) -print -quit | grep -q .; then
  tar -xzf "$R/sabdab2-test-pdb-20260721.tar.gz" -C "$P"
fi

test -s "$F/pairs.parquet"
test -s "$F/instances.parquet"
test -s "$F/splits.json"
test -s "$F/contamination.json"
test -s "$F/MANIFEST.json"

echo "manifest_sha256=$(sha256sum "$F/MANIFEST.json" | cut -d' ' -f1)"
echo "label_npz_count=$(find "$D/labels" -maxdepth 1 -name '*.npz' | wc -l)"
echo "mapping_json_count=$(find "$D/canonical/mapping" -maxdepth 1 -name '*.json' | wc -l)"
echo "structure_count=$(find "$P" -maxdepth 1 -type f \( -name '*.pdb' -o -name '*.cif' \) | wc -l)"
ls -lh "$F"
