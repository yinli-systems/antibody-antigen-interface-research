#!/bin/bash
#SBATCH --job-name=mipe_ind_test
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00

set -euo pipefail
umask 027

DEPLOY_ROOT=/ssd/scxi253/model_deployment
JOB_ROOT=$DEPLOY_ROOT/jobs/mipe_independent_test_v1
OUTPUT_ROOT=$DEPLOY_ROOT/results/mipe/official_independent_test_seed42_ensemble_v2
LOG_ROOT=$JOB_ROOT/logs
PYTHON_BIN=$DEPLOY_ROOT/envs/mipe/bin/python

mkdir -p "$LOG_ROOT"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite $OUTPUT_ROOT" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH
export TORCH_EXTENSIONS_DIR=${SLURM_TMPDIR:-/tmp}/mipe-torch-extensions-${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

"$PYTHON_BIN" "$JOB_ROOT/mipe_independent_test.py" \
  --run-root "$DEPLOY_ROOT/results/mipe/released_code_lr1e3_seed42_hardened_v3" \
  --source-root "$DEPLOY_ROOT/src/MIPE" \
  --evaluator-root "$DEPLOY_ROOT/jobs/mipe_evaluation/source_e7d190517a4c" \
  --test-data "$DEPLOY_ROOT/data/mipe/dataset/testdata.pkl" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda:0 \
  >"$LOG_ROOT/mipe-independent-test-${SLURM_JOB_ID}.out" \
  2>"$LOG_ROOT/mipe-independent-test-${SLURM_JOB_ID}.err"
