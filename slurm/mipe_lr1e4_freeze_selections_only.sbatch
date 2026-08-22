#!/usr/bin/env bash
#SBATCH --job-name=mipe_lr1e4_freeze
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH -o /ssd/scxi253/model_deployment/jobs/mipe_lr1e4_finalize_v1/logs/freeze-only-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/jobs/mipe_lr1e4_finalize_v1/logs/freeze-only-%j.err

set -euo pipefail
umask 027

readonly DEPLOY_ROOT=/ssd/scxi253/model_deployment
readonly FINAL_ROOT=${DEPLOY_ROOT}/jobs/mipe_lr1e4_finalize_v1
readonly RUN_ROOT=${DEPLOY_ROOT}/results/mipe/paper_appendix_lr1e4_seed42_fivefold_v3
readonly PYTHON=${DEPLOY_ROOT}/envs/contact/bin/python3.10

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-6}

# This job freezes development selections only. It must not read testdata.pkl or
# invoke mipe_independent_test.py while the held-out test gate remains closed.
"${PYTHON}" "${FINAL_ROOT}/freeze_mipe_selections.py" \
  --run-root "${RUN_ROOT}"
