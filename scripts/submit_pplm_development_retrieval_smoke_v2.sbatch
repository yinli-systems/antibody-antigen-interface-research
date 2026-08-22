#!/usr/bin/env bash
#SBATCH --job-name=pplm-dev-smoke-v2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/pplm-dev-smoke-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/pplm-dev-smoke-v2-%j.err

set -euo pipefail
umask 077

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]] || { echo "gpu_4090 is required" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 4090" ]] \
  || { echo "one RTX 4090 is required" >&2; exit 2; }

python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
data_root=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
result_root=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pplm_development_retrieval_smoke_v2
source_root=/ssd/scxi253/model_deployment/sources/pplm-1267db4592703095b889cc22da6cf1796ce038b5
implementation_root=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
model_root=/ssd/scxi253/model_deployment/models/pplm/revision-1267db4592703095b889cc22da6cf1796ce038b5
runner=${data_root}/tools/run_pplm_frozen_development_retrieval_v2.py
common_runner=${data_root}/tools/run_mint_frozen_development_retrieval.py
protocol=${data_root}/tools/abpairfm_pplm_development_retrieval_smoke_v2.json
runner_sha256=a2b715f00bfe05349978fe4512905a0b1a22b7272d9cedbb849c3b4beead58bd
common_runner_sha256=a8396ab4fd20e02d1f9e1cc90ea4361326fa1ab11044ecf923f33b322bab3a1e
protocol_sha256=2a2b64dae97e0f265580028461e0c03d0f064d6db6011abbf4c498a99f417578

[[ "$(sha256sum "${runner}" | cut -d" " -f1)" == "${runner_sha256}" ]] \
  || { echo "runner SHA-256 differs" >&2; exit 2; }
[[ "$(sha256sum "${common_runner}" | cut -d" " -f1)" == "${common_runner_sha256}" ]] \
  || { echo "common runner SHA-256 differs" >&2; exit 2; }
[[ "$(sha256sum "${protocol}" | cut -d" " -f1)" == "${protocol_sha256}" ]] \
  || { echo "protocol SHA-256 differs" >&2; exit 2; }

mkdir -p "${result_root}"
predictions=${result_root}/predictions.jsonl
summary=${result_root}/summary.json
[[ ! -e ${predictions} && ! -e ${summary} ]] \
  || { echo "refusing to overwrite PPLM smoke v2 outputs" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${python_bin}" "${runner}" \
  --dataset asep \
  --pair-panel "${data_root}/development_anchor/development_pairs_v1.jsonl" \
  --pair-panel-receipt "${data_root}/development_anchor/development_pairs_v1.receipt.json" \
  --protocol "${protocol}" \
  --protocol-sha256 "${protocol_sha256}" \
  --source-root "${source_root}" \
  --implementation-root "${implementation_root}" \
  --checkpoint "${model_root}/pplm_t33_650M.pt" \
  --head "${model_root}/ppi_models.pkl" \
  --head-audit /ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pair_retrieval_head_audit_v1.json \
  --predictions "${predictions}" \
  --summary "${summary}" \
  --maximum-batch-size 4 \
  --maximum-attention-cells 1048576 \
  --smoke-anchor-count 2
