#!/usr/bin/env bash
#SBATCH --job-name=fc-joint-v2-single
#SBATCH --partition=gpu_5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --chdir=/ssd/scxi253/model_deployment/data/framecontact_20260730/asep/joint-coupling-source-v2
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-v2-single-%A.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-joint-v2-single-%A.err

set -euo pipefail
: "${FC_TASK_INDEX:?FC_TASK_INDEX must identify one of the 30 preregistered development members}"
export FC_TASK_INDEX
exec /bin/bash /ssd/scxi253/model_deployment/data/framecontact_20260730/asep/joint-coupling-source-v2/slurm_train_framecontact_asep_joint_coupling_dev_v1.sbatch
