#!/usr/bin/env bash
#SBATCH --job-name=abf-late-scale-embed
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/embedding-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_late_stage_scaling_v1/logs/embedding-%A_%a.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_late_stage_scaling_v1
readonly tools=${root}/tools/abf30b_late_stage_scaling_v1
readonly protocol=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1.json
readonly protocol_receipt=${campaign}/authority/ABF30B_LATE_STAGE_SCALING_PROTOCOL_V1_RECEIPT.json
readonly base_campaign=${root}/results/development/abf30b_framecontact_v1
readonly chain_map=${base_campaign}/authority/SABDAB2_DEVELOPMENT_ABF30B_CHAIN_MAP_V1.json
readonly pretraining=${root}/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1
readonly pretrain_source=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/src
readonly family=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/protocols/antibody_model_family_v1.json
readonly sequence_view=${root}/data/framecontact_20260730/authorities/abpairfm_joint_embedding_v1/sabdab2_train_validation_sequences_v1.jsonl
readonly esm2_root=${root}/data/sppider_seq_sabdab2_embeddings
readonly python_bin=${root}/envs/contact/bin/python3.10

case "${SLURM_ARRAY_TASK_ID}" in
  0) readonly label=segment_000058; readonly segment=000058 ;;
  1) readonly label=segment_000063; readonly segment=000063 ;;
  *) exit 64 ;;
esac

readonly checkpoint=${pretraining}/segment-${segment}/run/checkpoint
readonly checkpoint_root=${campaign}/checkpoints/${label}
readonly authority=${checkpoint_root}/authority
readonly canary=${authority}/DCP_EMBEDDING_CANARY_V1.json
readonly canary_receipt=${authority}/DCP_EMBEDDING_CANARY_V1_RECEIPT.json
readonly embedding_root=${checkpoint_root}/embeddings/sabdab2_abf_antibody_esm2_antigen_v1

test "${SLURM_JOB_PARTITION}" = gpu_4090
test -n "${CUDA_VISIBLE_DEVICES:-}"
mkdir -p "${authority}"

read -r protocol_sha256 checkpoint_sha256 chain_map_sha256 < <(
  "${python_bin}" - "${protocol}" "${protocol_receipt}" "${label}" "${checkpoint}" <<'PY'
import hashlib,json,pathlib,sys
protocol_path, receipt_path, label, checkpoint_path = sys.argv[1:]
raw=open(protocol_path,'rb').read(); protocol=json.loads(raw)
receipt=json.load(open(receipt_path))
digest=hashlib.sha256(raw).hexdigest()
assert receipt['protocol_sha256']==digest
assert protocol['status']=='FROZEN_BEFORE_LATE_STAGE_SCALING_EXECUTION'
row=protocol['checkpoints'][label]
assert pathlib.Path(row['checkpoint_path']).resolve()==pathlib.Path(checkpoint_path).resolve()
assert label in protocol['execution']['new_embedding_checkpoints']
assert protocol['task_labels_read_during_freeze'] is False
assert protocol['test_loaded'] is False
assert protocol['temporal_holdout_loaded'] is False
print(digest,row['checkpoint_model_sha256'],protocol['chain_map_sha256'])
PY
)

printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${chain_map_sha256}" "${chain_map}" \
  f33e90400f6e7a55a434adbc72534fc7c971b7e60b946763c410287f1cc412c3 "${pretrain_source}/antibody_pretraining/model.py" \
  270a90b6c092a2890d0d84209578d4ba1e9b8c871c5377b57b5c72e3af24e274 "${pretrain_source}/antibody_pretraining/tokenizer.py" \
  | sha256sum --check --strict

export PYTHONPATH=${pretrain_source}
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if test ! -e "${canary}" && test ! -e "${canary_receipt}"; then
  "${python_bin}" "${tools}/canary_abf_checkpoint_v1.py" \
    --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
    --chain-map "${chain_map}" --chain-map-sha256 "${chain_map_sha256}" \
    --model-family "${family}" --checkpoint "${checkpoint}" \
    --checkpoint-label "${label}" --checkpoint-model-sha256 "${checkpoint_sha256}" \
    --output "${canary}" --receipt-output "${canary_receipt}"
fi

readonly canary_sha256=$("${python_bin}" - "${canary}" "${canary_receipt}" "${label}" "${checkpoint_sha256}" <<'PY'
import hashlib,json,sys
result_path, receipt_path, label, checkpoint_sha = sys.argv[1:]
raw=open(result_path,'rb').read(); result=json.loads(raw); receipt=json.load(open(receipt_path))
digest=hashlib.sha256(raw).hexdigest()
assert receipt['result_sha256']==digest
assert result['status']=='PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY'
assert result['checkpoint_label']==label
assert result['checkpoint_model_sha256']==checkpoint_sha
assert result['task_labels_read'] is False and result['test_loaded'] is False
print(digest)
PY
)

"${python_bin}" "${tools}/materialize_abf_checkpoint_sabdab2_embeddings_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
  --chain-map "${chain_map}" --chain-map-sha256 "${chain_map_sha256}" \
  --canary "${canary}" --canary-sha256 "${canary_sha256}" \
  --sequence-view "${sequence_view}" --model-family "${family}" \
  --checkpoint "${checkpoint}" --checkpoint-label "${label}" \
  --checkpoint-model-sha256 "${checkpoint_sha256}" \
  --esm2-root "${esm2_root}" --embedding-root "${embedding_root}" \
  --batch-antibodies 8 --resume
