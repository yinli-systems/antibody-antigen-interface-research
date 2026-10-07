#!/usr/bin/env bash
#SBATCH --job-name=abpairfm-pairloss-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pairloss-analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm-pairloss-analyze-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly PROJECT=${ROOT}/abpairfm_pair_loss_canary_v1/source
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly CAMPAIGN=${ROOT}/results/pretraining/abpairfm_pair_loss_canary_v1
readonly OUTPUT=${CAMPAIGN}/ANALYSIS_V1.json

test ! -e "${OUTPUT}"
"${PYTHON}" "${PROJECT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/analyze_pinder_pair_loss_canary_v1.py" \
  --campaign-root "${CAMPAIGN}" \
  --output "${OUTPUT}"
