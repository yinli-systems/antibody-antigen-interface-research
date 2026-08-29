#!/usr/bin/env bash
#SBATCH -J esmc-adapt-embed-full
#SBATCH -p gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-7%4
#SBATCH -o /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-full-%A_%a.out
#SBATCH -e /ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1/logs/adapted-embed-full-%A_%a.err

set -euo pipefail
readonly PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
readonly SOURCE=/ssd/scxi253/model_deployment/framecontact_high_value_three_stream_v1/source
readonly ROOT=/ssd/scxi253/model_deployment/results/prospective/framecontact_high_value_three_stream_v1/paired-esmc-cdr-v1
readonly EXECUTION=${ROOT}/authority/EXECUTION_ADAPTED_ESMC_EMBEDDING_FULL_V1.json
readonly SELF=${SOURCE}/slurm_adapted_esmc_embedding_full_v1.sbatch

readarray -t binding < <("${PYTHON}" - "${EXECUTION}" "${SELF}" <<'PY'
import hashlib, json, sys
from pathlib import Path

execution_path, launcher_path = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
value = json.loads(execution_path.read_text())
if (
    value.get("status") != "FROZEN_AFTER_CANARY_BEFORE_FULL_EMBEDDING"
    or value.get("authorization", {}).get("full_development_embedding_materialization_authorized") is not True
    or value.get("authorization", {}).get("same_protocol_downstream_development_training_authorized_before_full_audit") is not False
    or value.get("test_access") is not False
    or value.get("source_binding", {}).get("full_launcher_sha256") != sha(launcher_path)
):
    raise SystemExit("full adapted embedding execution authority differs")
print(value["parent_protocol"])
print(value["parent_protocol_sha256"])
print(value["development_sequence_plan"])
print(value["development_sequence_plan_sha256"])
print(value["outputs"]["result_root"])
print(value["outputs"]["whole_chain_adapted_root"])
print(value["outputs"]["cdr_focused_adapted_root"])
PY
)
readonly PROTOCOL=${binding[0]}
readonly PROTOCOL_SHA256=${binding[1]}
readonly PLAN=${binding[2]}
readonly PLAN_SHA256=${binding[3]}
readonly RESULT_ROOT=${binding[4]}
readonly WHOLE_ROOT=${binding[5]}
readonly CDR_ROOT=${binding[6]}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
[[ "${SLURM_JOB_PARTITION}" == gpu_4090 && "${gpu_name}" == "NVIDIA GeForce RTX 4090" ]]
if (( SLURM_ARRAY_TASK_ID < 4 )); then
  role=whole_chain_adapted
  shard=${SLURM_ARRAY_TASK_ID}
  checkpoint=${ROOT}/adaptation-production-v1/whole_chain_control/checkpoint-epoch-4.safetensors
  expected=50b27675df33f843e7a9a71eca4423c449603246980e3296dfbb21fc27ffa1fb
  output_root=${WHOLE_ROOT}
else
  role=cdr_focused_adapted
  shard=$((SLURM_ARRAY_TASK_ID - 4))
  checkpoint=${ROOT}/adaptation-production-v1/cdr_focused/checkpoint-epoch-4.safetensors
  expected=5fd67e390f4602f88e9a59d03a11bff332da487ebe4425bdb562bcce8a3a9d9a
  output_root=${CDR_ROOT}
fi
mkdir -p "${RESULT_ROOT}/${role}"
"${PYTHON}" "${SOURCE}/materialize_adapted_esmc_embeddings_v1.py" \
  --protocol "${PROTOCOL}" --protocol-sha256 "${PROTOCOL_SHA256}" \
  --plan "${PLAN}" --plan-sha256 "${PLAN_SHA256}" --role "${role}" \
  --checkpoint "${checkpoint}" --checkpoint-sha256 "${expected}" \
  --safetensors-site /ssd/scxi253/model_deployment/envs/epiformer/lib/python3.10/site-packages \
  --output-root "${output_root}" \
  --result "${RESULT_ROOT}/${role}/shard-${shard}.json" \
  --shard-index "${shard}" --shard-count 4
