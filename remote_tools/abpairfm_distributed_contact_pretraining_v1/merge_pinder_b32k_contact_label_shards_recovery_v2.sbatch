#!/usr/bin/env bash
#SBATCH --job-name=pinder-b32k-label-merge-r2
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-r2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-r2-%j.err

set -euo pipefail
readonly ROOT=/ssd/scxi253/model_deployment
readonly CURRICULUM=${ROOT}/data/abpairfm_3b_v1/pinder_contact_curriculum_v1
readonly SHARDS=${CURRICULUM}/B32K-contact-label-shards-recovery-v2
readonly PYTHON=${ROOT}/envs/contact/bin/python3.10
readonly SCRIPT=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/merge_pinder_contact_label_shards_v1.py
readonly BASE=${ROOT}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py

test "$(sha256sum "${SCRIPT}" | cut -d' ' -f1)" = "4c3a9c1900455f4b13c8a664903810db2ef481049c3a8ab56b5cba68a27c115b"
test "$(sha256sum "${BASE}" | cut -d' ' -f1)" = "a338fb9d3edd4c94154b73539176087d61a885b6faf5dcb17fb8936fd1091275"

"${PYTHON}" "${SCRIPT}" \
  --curriculum "${CURRICULUM}/B32K.tsv.gz" \
  --expected-curriculum-sha256 b79d0f086addb13b8596840268c34c5f7d51678981c51ed77c876219b7536386 \
  --curriculum-receipt "${CURRICULUM}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" \
  --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
  --shard-root "${SHARDS}" --shard-count 32 \
  --expected-shard-source-sha256 84c655f5cabaf6fa89e3f2fc874383d01670cf3d2da151d87273189b1814eda8 \
  --expected-base-source-sha256 a338fb9d3edd4c94154b73539176087d61a885b6faf5dcb17fb8936fd1091275 \
  --output "${CURRICULUM}/B32K-contact-labels-recovery-v2.jsonl.gz" \
  --receipt "${CURRICULUM}/B32K_CONTACT_LABELS_RECOVERY_V2_RECEIPT.json"
