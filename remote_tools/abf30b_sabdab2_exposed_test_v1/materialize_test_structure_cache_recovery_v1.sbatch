#!/usr/bin/env bash
#SBATCH --job-name=abf30b-test-structure-r1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/structure-recovery-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/exposed/abf30b_sabdab2_historical_test_v1/logs/structure-recovery-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/exposed/abf30b_sabdab2_historical_test_v1
readonly tools=${root}/tools/abf30b_sabdab2_exposed_test_v1
readonly protocol=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1.json
readonly receipt=${campaign}/authority/LABEL_FREE_TEST_PREPARATION_PLAN_V1_RECEIPT.json
readonly recovery=${campaign}/authority/STRUCTURE_EXEC_RECOVERY_PROTOCOL_V1.json
readonly recovery_receipt=${campaign}/authority/STRUCTURE_EXEC_RECOVERY_PROTOCOL_V1_RECEIPT.json
readonly input_root=${campaign}/label_free_inputs_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
test "${SLURM_JOB_PARTITION}" = gpu_4090
read -r protocol_sha recovery_sha < <("${python_bin}" - "${protocol}" "${receipt}" "${recovery}" "${recovery_receipt}" <<'PY'
import hashlib,json,sys
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
p,pr,r,rr=sys.argv[1:]; ps=sha(p); rs=sha(r); assert json.load(open(pr))['protocol_sha256']==ps; assert json.load(open(rr))['protocol_sha256']==rs; print(ps,rs)
PY
)
export PYTHONDONTWRITEBYTECODE=1
"${python_bin}" "${tools}/materialize_test_structure_cache_exec_recovery_v1.py" \
  --prep-protocol "${protocol}" --prep-protocol-sha256 "${protocol_sha}" \
  --exec-recovery-protocol "${recovery}" --exec-recovery-protocol-sha256 "${recovery_sha}" \
  --pinned-materializer "${root}/tools/sabdab2_predicted_geometry_v1/materialize_predicted_structure_cache.py" \
  --instance-manifest-csv /ssd/scxi253/yangwuyu/p_e_first_research/datasets/sabdab2/manifest.csv \
  --source-split-csv "${root}/data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv" \
  --paired-antibody-root /data/run01/scxi253/af3test/workdir/sabdab2_antibody_scxi5090_v2_5seed_20260716/output \
  --sdh-plan "${input_root}/TEST_INPUT_PLAN_V1.json" \
  --sdh-result-root "${input_root}/sdh_af3" \
  --antigen-root /data/run01/scxi253/af3test/workdir/sabdab2_antigen_scxi5090_v2_5seed_20260716/output \
  --af3-runner-script /data/run01/scxi253/fanxuezhe/projects/20260424_af3_qiangsheng/scripts/run_5090_singularity.sh \
  --output-root "${campaign}/data/test_structure_cache_v1"
