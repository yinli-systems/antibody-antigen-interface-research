#!/usr/bin/env bash
#SBATCH --job-name=pinder-b32k-label-merge-r3
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-r3-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-b32k-label-merge-r3-%j.err

set -euo pipefail
readonly R=/ssd/scxi253/model_deployment
readonly C=${R}/data/abpairfm_3b_v1/pinder_contact_curriculum_v1
readonly P=${R}/envs/contact/bin/python3.10
readonly X=${R}/remote_tools/abpairfm_distributed_contact_pretraining_v1/merge_pinder_contact_label_shards_v1.py
readonly B=${R}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py
test "$(sha256sum "${X}" | cut -d' ' -f1)" = 4c3a9c1900455f4b13c8a664903810db2ef481049c3a8ab56b5cba68a27c115b
test "$(sha256sum "${B}" | cut -d' ' -f1)" = 52c7182619cfe7fc5d3e47dfac9aabb49e2ae6e4507d5391ef3d4c879a8248c4
"${P}" "${X}" \
 --curriculum "${C}/B32K.tsv.gz" --expected-curriculum-sha256 b79d0f086addb13b8596840268c34c5f7d51678981c51ed77c876219b7536386 \
 --curriculum-receipt "${C}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
 --shard-root "${C}/B32K-contact-label-shards-recovery-v3" --shard-count 32 \
 --expected-shard-source-sha256 84c655f5cabaf6fa89e3f2fc874383d01670cf3d2da151d87273189b1814eda8 \
 --expected-base-source-sha256 52c7182619cfe7fc5d3e47dfac9aabb49e2ae6e4507d5391ef3d4c879a8248c4 \
 --output "${C}/B32K-contact-labels-recovery-v3.jsonl.gz" --receipt "${C}/B32K_CONTACT_LABELS_RECOVERY_V3_RECEIPT.json"
