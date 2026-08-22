#!/usr/bin/env bash
#SBATCH -J abpairfm-pplm650-audit
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH -o /ssd/scxi253/model_deployment/results/pretraining/abpairfm-pplm650-audit-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/pretraining/abpairfm-pplm650-audit-%j.err

set -euo pipefail

readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly PROJECT=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
readonly CHECKPOINT=/ssd/scxi253/model_deployment/models/pplm/revision-1267db4592703095b889cc22da6cf1796ce038b5/pplm_t33_650M.pt
readonly OUTPUT=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pplm_650m_checkpoint_compatibility_v2.json
readonly GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
readonly GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export PYTHONPATH="${PROJECT}/src"

if [[ "${GPU_COUNT}" -ne 1 || "${GPU_NAME}" != "NVIDIA GeForce RTX 4090" ]]; then
  echo "Forbidden PPLM audit allocation: count=${GPU_COUNT} name=${GPU_NAME}" >&2
  exit 2
fi
test -f "${CHECKPOINT}"
test ! -e "${OUTPUT}"
mkdir -p "$(dirname "${OUTPUT}")"
"${PYTHON}" "${PROJECT}/scripts/audit_pplm_650m_checkpoint_compatibility_v2.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-size 2604388437 \
  --expected-sha256 c8ce3b75dfd7b5890dfa7d96de7f698b522a5cbf59d7ee2732cb80ad3e5df495 \
  --source-revision 1267db4592703095b889cc22da6cf1796ce038b5 \
  --output "${OUTPUT}"
