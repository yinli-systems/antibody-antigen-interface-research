#!/bin/bash
#SBATCH --job-name=prosst-quant-official
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt4/logs/quant-%j.out
#SBATCH --error=/ssd/scxi253/frontier_multimodel_v1/structural_plm_attempt4/logs/quant-%j.err
set -euo pipefail
ROOT=/ssd/scxi253/frontier_multimodel_v1
A=$ROOT/structural_plm_attempt4
cd "$ROOT/source"
export PYTHONPATH="$ROOT/pylibs_prosst_official_v1:$ROOT/pylibs:$ROOT/external/ProSST:$PWD/src:$PWD" PYTHONHASHSEED=0
/ssd/scxi253/repr_extract/venv/bin/python3 - <<"PY"
import sklearn,numpy,scipy,joblib
assert (sklearn.__version__,numpy.__version__,scipy.__version__,joblib.__version__)==("1.2.2","1.26.4","1.12.0","1.3.2")
print("official quantizer environment",sklearn.__version__,numpy.__version__,scipy.__version__,joblib.__version__)
PY
/ssd/scxi253/repr_extract/venv/bin/python3 -u extensions/frontier_multimodel_v1/structural_plm.py prepare-prosst \
 --structure-units "$ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" --prosst-source "$ROOT/external/ProSST" \
 --prosst-source-commit 13bd56495641a84cd5475eaef7aa98b14fdc6ed5 --side antibody --shard-index 0 --shard-count 1 \
 --max-records 1 --device cuda --output "$A/prosst_tokens"
