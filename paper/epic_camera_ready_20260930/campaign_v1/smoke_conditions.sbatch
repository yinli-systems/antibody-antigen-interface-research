#!/usr/bin/env bash
#SBATCH --job-name=epic-rv-smoke
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
root=${1:?campaign root required}
protocol=${2:?campaign protocol required}
protocol_sha=${3:?campaign protocol sha required}
authority_receipt=${4:?authority receipt required}
authority_sha=${5:?authority receipt sha required}
index=${SLURM_ARRAY_TASK_ID:?array task required}
condition=$(python3 - "$protocol" "$index" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p["conditions"][int(sys.argv[2])]["name"])
PY
)
output="$root/smoke/$condition"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
exec /ssd/scxi253/model_deployment/envs/contact/bin/python3.10 "$root/source/train_condition.py" \
  --campaign-protocol "$protocol" \
  --campaign-protocol-sha256 "$protocol_sha" \
  --authority-receipt "$authority_receipt" \
  --authority-receipt-sha256 "$authority_sha" \
  --condition "$condition" --seed 17 --output "$output" --smoke
