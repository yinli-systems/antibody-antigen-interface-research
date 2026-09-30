#!/bin/bash
#SBATCH --job-name=prosst-extract-official
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt4/logs/extract-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt4/logs/extract-%j.err
set -euo pipefail
ROOT=/ssd/scxi253/frontier_multimodel_v1
A=$ROOT/structural_plm_attempt4
cd "$ROOT/source"
export PYTHONPATH="$ROOT/pylibs_prosst_official_v1:$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/structural_plm.py extract --model-kind prosst \
 --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" --tokens "$A/prosst_tokens" \
 --model-path "$ROOT/models/prosst2048" --model-revision e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8 \
 --layers 0 3 6 9 12 --side antibody --shard-index 0 --shard-count 1 --max-records 1 --output "$A/features"
