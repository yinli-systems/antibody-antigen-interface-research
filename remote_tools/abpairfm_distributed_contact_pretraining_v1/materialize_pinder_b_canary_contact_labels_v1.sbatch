#!/bin/bash
#SBATCH --job-name=pinder-bcanary-labels
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-bcanary-labels-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-bcanary-labels-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly DATA=${ROOT}/data/abpairfm_3b_v1
readonly RAW=${DATA}/raw/pinder_2024_02
readonly CURRICULUM=${DATA}/pinder_contact_curriculum_v1
readonly OUTPUT=${CURRICULUM}/B-canary-contact-labels-v1.jsonl.gz
readonly RECEIPT=${CURRICULUM}/B_CANARY_CONTACT_LABELS_V1_RECEIPT.json
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly SCRIPT=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py

mkdir -p "${ROOT}/logs"

"${PYTHON}" "${SCRIPT}" \
  --curriculum "${CURRICULUM}/B-canary.tsv.gz" \
  --expected-curriculum-sha256 603ce8101560fa2159785720be8adb9fc6f4a8b913137ff4241873bfbc91c6c4 \
  --curriculum-receipt "${CURRICULUM}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" \
  --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
  --pdb-archive "${RAW}/pdbs.zip" \
  --pdb-materialization-receipt "${RAW}/pdbs.materialization_v1.json" \
  --expected-pdb-materialization-receipt-sha256 a73721291fb8aa10a23909841095bd553f561613e0bc3356ba9a97bb127ec0bb \
  --index-parquet "${RAW}/index.parquet" \
  --expected-index-sha256 ced1936c4f36ea79161579df8d4b23bb4862b9c88f30af9defeeb85af99adac4 \
  --duckdb "${DATA}/tools/duckdb-1.5.4/duckdb" \
  --expected-duckdb-sha256 c478c0f49f6755100d55dee32de485c0f00f3729b40b62658a83b0666b684707 \
  --output "${OUTPUT}" \
  --receipt "${RECEIPT}" \
  --maximum-residues 1020
