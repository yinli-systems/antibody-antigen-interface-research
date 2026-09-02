#!/usr/bin/env bash
#SBATCH --job-name=abf30b-sable-embed
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-embed-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/development/abf30b_framecontact_v1/logs/sable-embed-%j.err

set -euo pipefail
readonly root=/ssd/scxi253/model_deployment
readonly campaign=${root}/results/development/abf30b_framecontact_v1
readonly sable=${root}/results/development/sable_v0_2_0_framecontact_external_v1
readonly authority=${campaign}/authority/sable_label_free_v1
readonly protocol=${authority}/ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1.json
readonly protocol_receipt=${authority}/ABF30B_SABLE_LABEL_FREE_PROTOCOL_V1_RECEIPT.json
readonly chain_map=${authority}/ABF30B_SABLE_CHAIN_MAP_V1.json
readonly chain_receipt=${authority}/ABF30B_SABLE_CHAIN_MAP_V1_RECEIPT.json
readonly canary=${campaign}/authority/ABF30B_DCP_EMBEDDING_CANARY_V1.json
readonly canary_receipt=${campaign}/authority/ABF30B_DCP_EMBEDDING_CANARY_V1_RECEIPT.json
readonly pretrain_source=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/src
readonly family=${root}/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1/protocols/antibody_model_family_v1.json
readonly checkpoint=${root}/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1/segment-000069/run/checkpoint
readonly structures=${sable}/structure_cache/predicted_label_free_v1/STRUCTURE_CACHE_MANIFEST.json
readonly control_embeddings=${sable}/embeddings/esm2_t33_650m_label_free_v1
readonly output=${campaign}/embeddings/sable_abf30b_antibody_esm2_antigen_v1
readonly python_bin=${root}/envs/contact/bin/python3.10

test "${SLURM_JOB_PARTITION}" = gpu_4090
read -r protocol_sha256 chain_map_sha256 canary_sha256 < <("${python_bin}" - "${protocol_receipt}" "${chain_receipt}" "${canary_receipt}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); a=json.load(open(sys.argv[3]))
assert p['frozen_before_mixed_embedding_materialization'] is True
assert c['status']=='PASS_LABEL_FREE_SABLE_ANARCII_CHAIN_TYPING'
assert a['status']=='PASS_DCP_EXACT_RESTORE_AND_RESIDUE_EMBEDDING_CANARY'
print(p['protocol_sha256'],c['chain_map_sha256'],a['result_sha256'])
PY
)
printf '%s  %s\n' \
  "${protocol_sha256}" "${protocol}" \
  "${chain_map_sha256}" "${chain_map}" \
  "${canary_sha256}" "${canary}" \
  795223b2ebd02d6ce37cebf7dadafe5b1f50b9b212976ba15e36af3e62f23259 "${structures}" \
  67670200608ac98573c9d0329fd06fb7f5973d0a47bd96aeea5861b94955abc9 "${control_embeddings}/SABLE_ESM_EMBEDDING_MANIFEST.json" \
  f33e90400f6e7a55a434adbc72534fc7c971b7e60b946763c410287f1cc412c3 "${pretrain_source}/antibody_pretraining/model.py" \
  270a90b6c092a2890d0d84209578d4ba1e9b8c871c5377b57b5c72e3af24e274 "${pretrain_source}/antibody_pretraining/tokenizer.py" \
  | sha256sum --check --strict
export PYTHONPATH=${pretrain_source}
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"${python_bin}" "${root}/tools/abf30b_framecontact_v1/materialize_abf30b_sable_embeddings_v1.py" \
  --protocol "${protocol}" --protocol-sha256 "${protocol_sha256}" \
  --chain-map "${chain_map}" --chain-map-sha256 "${chain_map_sha256}" \
  --canary "${canary}" --canary-sha256 "${canary_sha256}" \
  --structure-manifest "${structures}" \
  --control-embedding-root "${control_embeddings}" \
  --model-family "${family}" --checkpoint "${checkpoint}" \
  --output-root "${output}" --batch-antibodies 8
