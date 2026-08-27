#!/usr/bin/env bash
#SBATCH -J fc-temporal-open-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=24:00:00
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/open-%j.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/open-%j.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly DEP_SITE=/ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1
readonly AUTH=${ROOT}/authority
readonly LABEL_FREE=${ROOT}/TEMPORAL_DOUBLE_COLD_LABEL_FREE_MANIFEST_V1.json
readonly PARENT=${AUTH}/PROTOCOL_TEMPORAL_DOUBLE_COLD_V1.json
readonly PLAN=${AUTH}/TEMPORAL_EXECUTION_PLAN_V1.json
readonly PROTOCOL=${AUTH}/PROTOCOL_TEMPORAL_DOUBLE_COLD_MEASUREMENT_V1.json
readonly PROTOCOL_RECEIPT=${AUTH}/PROTOCOL_TEMPORAL_DOUBLE_COLD_MEASUREMENT_V1_RECEIPT.json
readonly FULL=/ssd/scxi253/model_deployment/results/development/sabdab2_edge_topk2_task_routed_full_v1/EDGE_TOPK2_TASK_ROUTED_FULL_DEVELOPMENT_ANALYSIS_V1.json
readonly BASE_SOURCE=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
readonly BASE_MANIFEST=/ssd/scxi253/model_deployment/results/development/sabdab2_matched_local_screen_v3/authority/SOURCE_MANIFEST_MATCHED_LOCAL_V3.json
readonly EXPERT_SOURCE=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1
readonly EXPERT_MANIFEST=${EXPERT_SOURCE}/SOURCE_MANIFEST_EDGE_TOPK2_V1.json
readonly ESM2=/ssd/scxi253/model_deployment/models/sppider_seq/esm2_t33_650M_UR50D
readonly DATA=${ROOT}/data-v1

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${SLURM_JOB_PARTITION}" == "gpu_4090" && "${gpu_count}" -eq 1 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
sha() { "${PYTHON}" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
label_free_sha=$(sha "${LABEL_FREE}")
"${PYTHON}" "${SOURCE}/freeze_temporal_measurement_v1.py" \
  --label-free-manifest "${LABEL_FREE}" \
  --label-free-manifest-sha256 "${label_free_sha}" \
  --parent-protocol "${PARENT}" \
  --parent-protocol-sha256 18df18a336e4a2107d6615329d08daec1ccc93790cf83132d1e75cb403b727bc \
  --execution-plan "${PLAN}" \
  --execution-plan-sha256 44a184b8e62986ef21abeff188f1014cd763e7cdd421a3177f12693f39000939 \
  --full-analysis "${FULL}" \
  --full-analysis-sha256 f4368275e38d5134e667e23470702e987bb1a301b1c89aee5e82718d39911824 \
  --baseline-source-root "${BASE_SOURCE}" \
  --baseline-source-manifest "${BASE_MANIFEST}" \
  --baseline-source-manifest-sha256 f649fb5a16cda6ff79eae05b2d2a3fedd6ffd02bd7b0b1cba1f1d0385a9a2f81 \
  --expert-source-root "${EXPERT_SOURCE}" \
  --expert-source-manifest "${EXPERT_MANIFEST}" \
  --expert-source-manifest-sha256 f09790fdb389bf88c248d1936f8340956ec3692cea8269af8ee4a9716aa2fa55 \
  --esm2-model-root "${ESM2}" \
  --output "${PROTOCOL}" \
  --receipt "${PROTOCOL_RECEIPT}"
protocol_sha=$(sha "${PROTOCOL}")
PYTHONPATH="${DEP_SITE}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  "${SOURCE}/materialize_temporal_double_cold_v1.py" \
  --protocol "${PROTOCOL}" \
  --protocol-sha256 "${protocol_sha}" \
  --source-root "${EXPERT_SOURCE}" \
  --model-root "${ESM2}" \
  --output-root "${DATA}"
