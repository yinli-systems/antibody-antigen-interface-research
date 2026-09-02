#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-analyze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/analyze-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/analyze-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly protocol=${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json
readonly receipt=${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha=$("${python_bin}" - "${protocol}" "${receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); r=json.load(open(sys.argv[2])); d=hashlib.sha256(raw).hexdigest(); assert r['protocol_sha256']==d; print(d)
PY
)
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${root}/tools/abf30b_sabdab2_exposed_test_v1/analyze_exposed_test_v1.py" \
  --campaign-root "${campaign}" --protocol-sha256 "${protocol_sha}" \
  --test-panel "${root}/data/framecontact_20260730/partner_metadata_antigen_fold_v3/test-reciprocal-31.json"
