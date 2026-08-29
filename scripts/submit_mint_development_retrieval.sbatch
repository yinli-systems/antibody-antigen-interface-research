#!/usr/bin/env bash
#SBATCH --job-name=mint-dev-ret
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/mint-dev-ret-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/mint-dev-ret-%A_%a.err

set -euo pipefail
umask 077

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]] || { echo "gpu_4090 is required" >&2; exit 2; }
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 4090" ]] \
  || { echo "one RTX 4090 is required" >&2; exit 2; }

python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
data_root=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
result_root=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/mint_development_retrieval_v1
source_root=/ssd/scxi253/model_deployment/sources/mint-06694b7606e2d00b76ec58daf5c7aecdaf7cd283
model_root=/ssd/scxi253/model_deployment/models/mint/revision-75b1602c34e9df1a89089a1245f000a51ee3c746
runner=${data_root}/tools/run_mint_frozen_development_retrieval.py
protocol=${data_root}/tools/abpairfm_mint_development_retrieval_v1.json
protocol_sha256=aed09ef940806e2599b55905708e6b0d933e0c2bbf2803ef741d9883ad51a25f
runner_sha256=a8396ab4fd20e02d1f9e1cc90ea4361326fa1ab11044ecf923f33b322bab3a1e
smoke_anchor_count=${MINT_SMOKE_ANCHOR_COUNT:-0}

case ${SLURM_ARRAY_TASK_ID} in
  0) dataset=asep ;;
  1) dataset=sabdab2 ;;
  *) echo "unsupported array task" >&2; exit 2 ;;
esac

[[ "$(sha256sum "${runner}" | cut -d" " -f1)" == "${runner_sha256}" ]] \
  || { echo "runner SHA-256 differs" >&2; exit 2; }
[[ "$(sha256sum "${protocol}" | cut -d" " -f1)" == "${protocol_sha256}" ]] \
  || { echo "protocol SHA-256 differs" >&2; exit 2; }

if (( smoke_anchor_count > 0 )); then
  output_root=${result_root}/smoke/${dataset}-anchors-${smoke_anchor_count}
else
  output_root=${result_root}/${dataset}
fi
mkdir -p "${output_root}"
[[ ! -e ${output_root}/predictions.jsonl && ! -e ${output_root}/summary.json ]] \
  || { echo "refusing to overwrite retrieval outputs" >&2; exit 2; }

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${python_bin}" "${runner}" \
  --dataset "${dataset}" \
  --pair-panel "${data_root}/development_anchor/development_pairs_v1.jsonl" \
  --pair-panel-receipt "${data_root}/development_anchor/development_pairs_v1.receipt.json" \
  --protocol "${protocol}" \
  --protocol-sha256 "${protocol_sha256}" \
  --source-root "${source_root}" \
  --checkpoint "${model_root}/mint.ckpt" \
  --head "${model_root}/bernett_mlp.pth" \
  --head-audit /ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pair_retrieval_head_audit_v1.json \
  --predictions "${output_root}/predictions.jsonl" \
  --summary "${output_root}/summary.json" \
  --maximum-batch-size 8 \
  --maximum-attention-cells 4194304 \
  --smoke-anchor-count "${smoke_anchor_count}"
