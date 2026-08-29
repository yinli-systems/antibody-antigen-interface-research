#!/bin/bash
#SBATCH --job-name=mipe_lr1e4_eval
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00

set -euo pipefail
umask 027

DEPLOY_ROOT=/ssd/scxi253/model_deployment
FINAL_ROOT=$DEPLOY_ROOT/jobs/mipe_lr1e4_finalize_v1
RUN_ROOT=$DEPLOY_ROOT/results/mipe/paper_appendix_lr1e4_seed42_fivefold_v3
OUTPUT_ROOT=$DEPLOY_ROOT/results/mipe/official_independent_test_paper_lr1e4_seed42_ensemble_v1
LOG_ROOT=$FINAL_ROOT/logs
mkdir -p "$LOG_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH
export TORCH_EXTENSIONS_DIR=${SLURM_TMPDIR:-/tmp}/mipe-lr1e4-eval-ext-${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

$DEPLOY_ROOT/envs/contact/bin/python3.10 "$FINAL_ROOT/freeze_mipe_selections.py" \
  --run-root "$RUN_ROOT" \
  >"$LOG_ROOT/freeze-${SLURM_JOB_ID}.out" \
  2>"$LOG_ROOT/freeze-${SLURM_JOB_ID}.err"

$DEPLOY_ROOT/envs/mipe/bin/python "$FINAL_ROOT/mipe_independent_test.py" \
  --run-root "$RUN_ROOT" \
  --source-root "$DEPLOY_ROOT/src/MIPE" \
  --evaluator-root "$DEPLOY_ROOT/jobs/mipe_evaluation/source_e7d190517a4c" \
  --test-data "$DEPLOY_ROOT/data/mipe/dataset/testdata.pkl" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda:0 \
  >"$LOG_ROOT/eval-${SLURM_JOB_ID}.out" \
  2>"$LOG_ROOT/eval-${SLURM_JOB_ID}.err"
