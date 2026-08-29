#!/bin/bash
#SBATCH --job-name=mipe_lr1e4
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=30:00:00
#SBATCH --signal=B:USR1@900

set -euo pipefail
umask 027

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/contact/bin/python3.10
HARDENED_ROOT=$DEPLOY_ROOT/jobs/mipe_paper_lr1e4_v3
SOURCE_ROOT=$DEPLOY_ROOT/src/MIPE
DATA_PATH=$DEPLOY_ROOT/data/mipe/dataset/cvdata.pkl
RUN_ROOT=$DEPLOY_ROOT/results/mipe/paper_appendix_lr1e4_seed42_fivefold_v3
FOLD=${MIPE_FOLD:?MIPE_FOLD must be in the range 0 through 4}
INVOCATION_ID=slurm-${SLURM_JOB_ID}-fold-${FOLD}
FOLD_ROOT=$RUN_ROOT/fold_${FOLD}

if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "MIPE_FOLD must be in the range 0 through 4" >&2
  exit 2
fi
if [[ ! -f "$RUN_ROOT/run_manifest.json" ]]; then
  echo "Fresh paper-protocol run is not bootstrapped" >&2
  exit 2
fi

LOCAL_BASE=${SLURM_TMPDIR:-/tmp}
JOB_LOCAL_ROOT=$LOCAL_BASE/mipe-paper-lr1e4-${SLURM_JOB_ID}-fold-${FOLD}
EXTENSION_ROOT=$JOB_LOCAL_ROOT/torch_extensions
mkdir -p "$JOB_LOCAL_ROOT/tmp" "$EXTENSION_ROOT"
export TMPDIR=$JOB_LOCAL_ROOT/tmp
export TORCH_EXTENSIONS_DIR=$EXTENSION_ROOT
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$DEPLOY_ROOT/envs/mipe/lib/python3.10/site-packages
export PATH=$DEPLOY_ROOT/envs/mipe/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

LOCAL_LOG=$JOB_LOCAL_ROOT/training.log
FINAL_LOG=$FOLD_ROOT/logs/${INVOCATION_ID}.log
exec 9>"$FOLD_ROOT/.continuation.lock"
if ! flock -n 9; then
  echo "Another training process owns fold $FOLD" >&2
  exit 3
fi

copy_log() {
  local status=$?
  if [[ -f "$LOCAL_LOG" ]]; then
    cp "$LOCAL_LOG" "$FINAL_LOG.tmp.${SLURM_JOB_ID}"
    chmod 0444 "$FINAL_LOG.tmp.${SLURM_JOB_ID}"
    mv "$FINAL_LOG.tmp.${SLURM_JOB_ID}" "$FINAL_LOG"
  fi
  return "$status"
}
TRAIN_PID=
forward_usr1() {
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -USR1 "$TRAIN_PID"
  fi
}
trap copy_log EXIT
trap forward_usr1 USR1

cd "$JOB_LOCAL_ROOT"
"$PYTHON_BIN" -S "$HARDENED_ROOT/continue_train.py" \
  --run-root "$RUN_ROOT" \
  --source-root "$SOURCE_ROOT" \
  --data "$DATA_PATH" \
  --hardened-source-root "$HARDENED_ROOT" \
  --submission-script "$HARDENED_ROOT/mipe_paper_lr1e4.sbatch" \
  --fold "$FOLD" \
  --invocation-id "$INVOCATION_ID" \
  --learning-rate 0.0001 \
  --validation-interval 1 \
  --graph-cache off \
  --maximum-wall-seconds 103500 \
  >"$LOCAL_LOG" 2>&1 &
TRAIN_PID=$!
wait "$TRAIN_PID"
