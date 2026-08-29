#!/usr/bin/env bash
#SBATCH -J esmc-adapt-canary-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adaptation-canary-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adaptation-canary-%A_%a.err

set -euo pipefail
: "${EXPECTED_PROTOCOL_SHA256:?submit with frozen protocol SHA-256}"
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly AUTH=${ROOT}/authority/PROTOCOL_ESMC_ADAPTATION_CANARY_V1.json
readonly CHECKPOINT=${ROOT}/models/biohub--ESMC-600M/a7e82012c83126b9eedb055fea9fa84b6c02f094/model.safetensors
readonly CORPUS=/ssd/scxi253/model_deployment/data/abpairfm_3b_v1/normalized_candidates/oas_paired_20260821/oas_stage_b_devclean_protocol_v3_unordered_v2.tsv.gz
readonly SPANS=${ROOT}/corpus/paired_esmc_cdr_spans_and_split_v5.tsv.gz
test "${SLURM_JOB_PARTITION}" = gpu_4090
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)" = "NVIDIA GeForce RTX 4090"
printf '%s  %s\n' "${EXPECTED_PROTOCOL_SHA256}" "${AUTH}" | sha256sum --check --strict
if [[ "${SLURM_ARRAY_TASK_ID}" == 0 ]]; then arm=whole_chain_control; else arm=cdr_focused; fi
readonly OUTPUT=${ROOT}/adaptation-canary-v1/${arm}.json
test ! -e "${OUTPUT}"
export PYTHONPATH=${SOURCE}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${SOURCE}/run_esmc_adaptation_canary_v1.py" \
  --authority "${AUTH}" --checkpoint "${CHECKPOINT}" --source "${CORPUS}" --spans "${SPANS}" \
  --arm "${arm}" --safetensors-site /ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages \
  --output "${OUTPUT}"
