#!/usr/bin/env bash
#SBATCH --job-name=abf-late-scale-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/analyze-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_late_stage_scaling_v1
readonly tools=${root}/tools/abf30b_late_stage_scaling_v1
readonly protocol=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1.json
readonly protocol_receipt=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1_RECEIPT.json
readonly control=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1/PREDICTED_GEOMETRY_NO_PRIVILEGED_DISTANCE_ANALYSIS_V1.json
readonly final_root=${root}/results/development/abf30b_framecontact_v1
readonly final_analysis=${final_root}/ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1.json
readonly final_receipt=${final_root}/ABF30B_FRAMECONTACT_DEVELOPMENT_ANALYSIS_V1_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha256=$("${python_bin}" - "${protocol}" "${protocol_receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); receipt=json.load(open(sys.argv[2]))
digest=hashlib.sha256(raw).hexdigest(); assert receipt['protocol_sha256']==digest
print(digest)
PY
)
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools}/analyze_late_stage_scaling_v1.py" \
  --campaign-root "${campaign}" \
  --scaling-protocol "${protocol}" --scaling-protocol-sha256 "${protocol_sha256}" \
  --control-analysis "${control}" \
  --final-analysis "${final_analysis}" --final-analysis-receipt "${final_receipt}" \
  --output "${campaign}/ABF30B_LATE_STAGE_SCALING_ANALYSIS_V1.json" \
  --receipt-output "${campaign}/ABF30B_LATE_STAGE_SCALING_ANALYSIS_V1_RECEIPT.json"
