#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-inputs
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/inputs-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/inputs-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly protocol=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json
readonly receipt=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
readonly protocol_sha=$("${python_bin}" - "${protocol}" "${receipt}" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); r=json.load(open(sys.argv[2])); d=hashlib.sha256(raw).hexdigest(); assert r['protocol_sha256']==d; print(d)
PY
)
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools}/build_test_inputs_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha}" \
  --split "${root}/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv" \
  --instance-manifest /ssd/scxi253/yangwuyu/p_e_first_research/datasets/sabdab2/manifest.csv \
  --paired-root /data/run01/scxi253/af3test/workdir/sabdab2_antibody_scxi5090_v2_5seed_20260716/output \
  --antigen-root /data/run01/scxi253/af3test/workdir/sabdab2_antigen_scxi5090_v2_5seed_20260716/output \
  --output-root "${campaign}/label_free_inputs_v1" --batch-size 20
