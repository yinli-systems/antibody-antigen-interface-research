#!/usr/bin/env bash
#SBATCH --job-name=pinder-b32k-label-merge
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-%j.err

set -euo pipefail

readonly ROOT=/ssd/scxi253/model_deployment
readonly DATA=${ROOT}/data/abpairfm_3b_v1
readonly CURRICULUM=${DATA}/pinder_contact_curriculum_v1
readonly SHARDS=${CURRICULUM}/B32K-contact-label-shards-v1
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly SCRIPT=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/merge_pinder_contact_label_shards_v1.py
readonly BASE=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py

test "$(sha256sum "${SCRIPT}" | cut -d' ' -f1)" = "49fc8fa41916495f6592bd08f7575b2297f02b60cf435dbf2435566b72325610"
test "$(sha256sum "${BASE}" | cut -d' ' -f1)" = "5c02f12d1b153e52e9fde35cefb1f92949893cc10e594a8f3bb02204823f8f4e"

"${PYTHON}" "${SCRIPT}" \
  --curriculum "${CURRICULUM}/B32K.tsv.gz" \
  --expected-curriculum-sha256 b79d0f086addb13b8596840268c34c5f7d51678981c51ed77c876219b7536386 \
  --curriculum-receipt "${CURRICULUM}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" \
  --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
  --shard-root "${SHARDS}" \
  --shard-count 32 \
  --output "${CURRICULUM}/B32K-contact-labels-v1.jsonl.gz" \
  --receipt "${CURRICULUM}/B32K_CONTACT_LABELS_V1_RECEIPT.json"
