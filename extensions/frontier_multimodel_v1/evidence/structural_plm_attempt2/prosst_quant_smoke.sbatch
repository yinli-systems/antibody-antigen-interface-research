#!/bin/bash
#SBATCH --job-name=prosst-quant-smoke2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt2/logs/prosst-quant-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt2/logs/prosst-quant-%j.err
set -euo pipefail
ROOT=/ssd/scxi253/frontier_multimodel_v1; cd "$ROOT/source"
export PYTHONPATH="$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/structural_plm.py prepare-prosst \
 --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" --prosst-source "$ROOT/external/ProSST" \
 --prosst-source-commit 13bd56495641a84cd5475eaef7aa98b14fdc6ed5 --side antibody --shard-index 0 --shard-count 1 \
 --max-records 1 --device cuda --output "$A/prosst_tokens"
