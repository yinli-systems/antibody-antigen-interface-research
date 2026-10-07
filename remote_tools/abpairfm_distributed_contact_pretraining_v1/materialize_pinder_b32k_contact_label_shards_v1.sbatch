#!/usr/bin/env bash
#SBATCH --job-name=pinder-b32k-labels
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-31%8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-b32k-labels-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-b32k-labels-%A_%a.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly DATA=${ROOT}/data/abpairfm_3b_v1
readonly RAW=${DATA}/raw/pinder_2024_02
readonly CURRICULUM=${DATA}/pinder_contact_curriculum_v1
readonly SHARDS=${CURRICULUM}/B32K-contact-label-shards-v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly SCRIPT=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_label_shard_v1.py
readonly BASE=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py
readonly PART=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")

test "$(sha256sum "${SCRIPT}" | cut -d' ' -f1)" = "19ee2338c4f8ec2eab44fcc0738a61c7d4e6be1bee05f3c40ab87fc0b55e414b"
test "$(sha256sum "${BASE}" | cut -d' ' -f1)" = "5c02f12d1b153e52e9fde35cefb1f92949893cc10e594a8f3bb02204823f8f4e"
mkdir -p "${ROOT}/logs" "${SHARDS}"
test ! -e "${SHARDS}/part-${PART}.jsonl.gz"
test ! -e "${SHARDS}/part-${PART}.receipt.json"

"${PYTHON}" "${SCRIPT}" \
  --curriculum "${CURRICULUM}/B32K.tsv.gz" \
  --expected-curriculum-sha256 b79d0f086addb13b8596840268c34c5f7d51678981c51ed77c876219b7536386 \
  --curriculum-receipt "${CURRICULUM}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" \
  --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
  --pdb-archive "${RAW}/pdbs.zip" \
  --pdb-materialization-receipt "${RAW}/pdbs.materialization_v1.json" \
  --expected-pdb-materialization-receipt-sha256 a73721291fb8aa10a23909841095bd553f561613e0bc3356ba9a97bb127ec0bb \
  --index-parquet "${RAW}/index.parquet" \
  --expected-index-sha256 ced1936c4f36ea79161579df8d4b23bb4862b9c88f30af9defeeb85af99adac4 \
  --duckdb "${DATA}/tools/duckdb-1.5.4/duckdb" \
  --expected-duckdb-sha256 c478c0f49f6755100d55dee32de485c0f00f3729b40b62658a83b0666b684707 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-count 32 \
  --output "${SHARDS}/part-${PART}.jsonl.gz" \
  --receipt "${SHARDS}/part-${PART}.receipt.json" \
  --maximum-residues 1020
