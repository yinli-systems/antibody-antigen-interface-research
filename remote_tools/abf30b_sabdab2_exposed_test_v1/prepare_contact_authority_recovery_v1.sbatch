#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-contact-adapter
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/contact-adapter-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/contact-adapter-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly protocol=${campaign}/authority/EXPOSED_TEST_MEASUREMENT_PROTOCOL_V1.json
readonly split_recovery=${campaign}/authority/TEST_SPLIT_VIEW_EXECUTION_RECOVERY_V1.json
readonly adapter_recovery=${campaign}/authority/CONTACT_AUTHORITY_ADAPTER_RECEIPT_RECOVERY_V1.json
readonly adapter_recovery_receipt=${campaign}/authority/CONTACT_AUTHORITY_ADAPTER_RECEIPT_RECOVERY_V1_RECEIPT.json
readonly adapter_root=${campaign}/data/test_contact_authority_adapter_v1
readonly evaluation_recovery=${campaign}/authority/CONTACT_AUTHORITY_EVALUATION_RECOVERY_V1.json
readonly evaluation_recovery_receipt=${campaign}/authority/CONTACT_AUTHORITY_EVALUATION_RECOVERY_V1_RECEIPT.json
read -r adapter_recovery_sha < <("${python_bin}" - "${adapter_recovery}" "${adapter_recovery_receipt}" <<'PY'
import hashlib,json,sys
p,r=sys.argv[1:]
digest=hashlib.sha256(open(p,'rb').read()).hexdigest()
assert json.load(open(r))['recovery_sha256']==digest
print(digest)
PY
)
test "${SLURM_JOB_PARTITION}" = gpu_4090
test -n "${CUDA_VISIBLE_DEVICES:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" "${tools}/materialize_test_contact_authority_adapter_v1.py" \
  --recovery-protocol "${adapter_recovery}" \
  --recovery-protocol-sha256 "${adapter_recovery_sha}" \
  --source-contact-root "${root}/data/framecontact_20260730/sabdab2/contact_cohort_v2" \
  --structure-root "${campaign}/data/test_structure_cache_v1" \
  --output-root "${adapter_root}"
"${python_bin}" "${tools}/freeze_contact_authority_evaluation_recovery_v1.py" \
  --measurement-protocol "${protocol}" \
  --split-recovery "${split_recovery}" \
  --adapter-recovery "${adapter_recovery}" \
  --adapter-recovery-receipt "${adapter_recovery_receipt}" \
  --adapter-root "${adapter_root}" \
  --source "${tools}/evaluate_exposed_test_contact_authority_recovery_v1.py" \
  --source "${tools}/evaluate_exposed_test_contact_authority_recovery_v1.sbatch" \
  --source "${tools}/analyze_exposed_test_v1.py" \
  --source "${tools}/analyze_exposed_test_v1.sbatch" \
  --output "${evaluation_recovery}" \
  --receipt-output "${evaluation_recovery_receipt}"
