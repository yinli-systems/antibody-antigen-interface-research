#!/usr/bin/env bash
#SBATCH --job-name=pplm-dev-ret
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/pplm-dev-ret-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/logs/pplm-dev-ret-%A_%a.err

set -euo pipefail
umask 077

[[ ${SLURM_JOB_PARTITION:-} == gpu_4090 ]] || exit 2
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" == "NVIDIA GeForce RTX 4090" ]] || exit 2

python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
data_root=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1
result_root=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pplm_development_retrieval_v1
source_root=/ssd/scxi253/model_deployment/sources/pplm-1267db4592703095b889cc22da6cf1796ce038b5
implementation_root=/ssd/scxi253/model_deployment/abpairfm_3b_v1/source
model_root=/ssd/scxi253/model_deployment/models/pplm/revision-1267db4592703095b889cc22da6cf1796ce038b5
runner=${data_root}/tools/run_pplm_frozen_development_retrieval_v5.py
common_runner=${data_root}/tools/run_mint_frozen_development_retrieval.py
protocol=${data_root}/tools/abpairfm_pplm_development_retrieval_v1.json
smoke_summary=/ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pplm_development_retrieval_smoke_v4/summary.json
runner_sha256=28dfadf396fdd0c661aa280c74c7800556c8857bdb90972890e663144faafd3d
common_runner_sha256=a8396ab4fd20e02d1f9e1cc90ea4361326fa1ab11044ecf923f33b322bab3a1e
protocol_sha256=8b0d9332aa1ca84ec578aa1b685dfc4b187856dde94f0d6ecec8d64978875071
smoke_summary_sha256=1f642e37d529834644774ec627d6d4f651916f51ceb631022a258690551ad47d

[[ "$(sha256sum "${runner}" | cut -d" " -f1)" == "${runner_sha256}" ]] || exit 2
[[ "$(sha256sum "${common_runner}" | cut -d" " -f1)" == "${common_runner_sha256}" ]] || exit 2
[[ "$(sha256sum "${protocol}" | cut -d" " -f1)" == "${protocol_sha256}" ]] || exit 2
[[ "$(sha256sum "${smoke_summary}" | cut -d" " -f1)" == "${smoke_summary_sha256}" ]] || exit 2

case ${SLURM_ARRAY_TASK_ID} in
  0) dataset=asep ;;
  1) dataset=sabdab2 ;;
  *) exit 2 ;;
esac

output_root=${result_root}/${dataset}
mkdir -p "${output_root}"
predictions=${output_root}/predictions.jsonl
summary=${output_root}/summary.json
[[ ! -e ${predictions} && ! -e ${summary} ]] || exit 2

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
  --implementation-root "${implementation_root}" \
  --checkpoint "${model_root}/pplm_t33_650M.pt" \
  --head "${model_root}/ppi_models.pkl" \
  --head-audit /ssd/scxi253/model_deployment/results/pretraining/abpairfm_3b_v1/baselines/pair_retrieval_head_audit_v1.json \
  --equivalence-smoke-summary "${smoke_summary}" \
  --predictions "${predictions}" \
  --summary "${summary}" \
  --maximum-batch-size 4 \
  --maximum-attention-cells 1048576
