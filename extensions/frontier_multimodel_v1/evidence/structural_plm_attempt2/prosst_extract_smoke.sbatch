#!/bin/bash
#SBATCH --job-name=prosst-real-smoke2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt2/logs/prosst-extract-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt2/logs/prosst-extract-%j.err
set -euo pipefail
ROOT=/ssd/scxi253/frontier_multimodel_v1; cd "$ROOT/source"
export PYTHONPATH="$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0
[[ $(sha256sum "$ROOT/models/prosst2048/model.safetensors" | cut -d" " -f1) == aa6742cd2fd4c0ad3b908f19358136c66614f07093cdf9a995120723c6d54368 ]]
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/structural_plm.py extract --model-kind prosst \
 --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" --tokens "$A/prosst_tokens" \
 --model-path "$ROOT/models/prosst2048" --model-revision e94ffee7846d7f55c1bf5efa8ec7372a336ac4b8 \
 --layers 0 3 6 9 12 --side antibody --shard-index 0 --shard-count 1 --max-records 1 --output "$A/features"
