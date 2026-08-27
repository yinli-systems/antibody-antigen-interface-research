#!/usr/bin/env bash
#SBATCH -J fc-temporal-eval-v1
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/eval-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1/logs/eval-%A_%a.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/temporal-double-cold-v1
readonly PROTOCOL=${ROOT}/authority/PROTOCOL_TEMPORAL_DOUBLE_COLD_MEASUREMENT_V1.json
readonly DATA=${ROOT}/data-v1
readonly COMPONENTS=${ROOT}/components-v1
readonly BASE_SOURCE=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/matched_local_training_source_v3
readonly EXPERT_SOURCE=/ssd/scxi253/model_deployment/immutable_stage_a1_candidates/edge_topk2_training_source_v1
readonly SEEDS=(17 29 43)

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[[ "${SLURM_JOB_PARTITION}" == "gpu_4090" && "${gpu_count}" -eq 1 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
sha() { "${PYTHON}" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
protocol_sha=$(sha "${PROTOCOL}")
data_sha=$(sha "${DATA}/TEMPORAL_DOUBLE_COLD_DATA_MANIFEST_V1.json")
seed=${SEEDS[${SLURM_ARRAY_TASK_ID}]}
for arm in baseline_current edge_topk2_supervised; do
  if [[ "${arm}" == baseline_current ]]; then
    arm_source=${BASE_SOURCE}
  else
    arm_source=${EXPERT_SOURCE}
  fi
  out=${COMPONENTS}/${arm}/seed-${seed}
  mkdir -p "${out}"
  "${PYTHON}" "${SOURCE}/evaluate_temporal_double_cold_component_v1.py" \
    --protocol "${PROTOCOL}" \
    --protocol-sha256 "${protocol_sha}" \
    --data-root "${DATA}" \
    --data-manifest-sha256 "${data_sha}" \
    --arm "${arm}" \
    --seed "${seed}" \
    --source-root "${arm_source}" \
    --output "${out}/RESULT.json" \
    --receipt "${out}/RECEIPT.json"
done
