#!/usr/bin/env bash
#SBATCH --job-name=pinder-b32k-labels-r3
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --array=0-31%8
#SBATCH --time=02:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/pinder-b32k-labels-r3-%A_%a.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/pinder-b32k-labels-r3-%A_%a.err

set -euo pipefail
readonly R=/ssd/scxi253/model_deployment
readonly D=${R}/data/abpairfm_3b_v1
readonly C=${D}/pinder_contact_curriculum_v1
readonly S=${C}/B32K-contact-label-shards-recovery-v3
readonly P=${R}/envs/contact/bin/python3.10
readonly X=${R}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_label_shard_v1.py
readonly B=${R}/remote_tools/abpairfm_distributed_contact_pretraining_v1/materialize_pinder_contact_labels_v1.py
readonly I=$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")
test "$(sha256sum "${X}" | cut -d' ' -f1)" = 84c655f5cabaf6fa89e3f2fc874383d01670cf3d2da151d87273189b1814eda8
test "$(sha256sum "${B}" | cut -d' ' -f1)" = 52c7182619cfe7fc5d3e47dfac9aabb49e2ae6e4507d5391ef3d4c879a8248c4
mkdir -p "${R}/logs" "${S}"
test ! -e "${S}/part-${I}.jsonl.gz"; test ! -e "${S}/part-${I}.receipt.json"
"${P}" "${X}" \
 --curriculum "${C}/B32K.tsv.gz" --expected-curriculum-sha256 b79d0f086addb13b8596840268c34c5f7d51678981c51ed77c876219b7536386 \
 --curriculum-receipt "${C}/PINDER_CONTACT_CURRICULUM_RECEIPT_V1.json" --expected-curriculum-receipt-sha256 dc44cb732535aeaefb660794c2f4294226ef569e73c96223261decf4465dae0a \
 --pdb-archive "${D}/raw/pinder_2024_02/pdbs.zip" --pdb-materialization-receipt "${D}/raw/pinder_2024_02/pdbs.materialization_v1.json" --expected-pdb-materialization-receipt-sha256 a73721291fb8aa10a23909841095bd553f561613e0bc3356ba9a97bb127ec0bb \
 --index-parquet "${D}/raw/pinder_2024_02/index.parquet" --expected-index-sha256 ced1936c4f36ea79161579df8d4b23bb4862b9c88f30af9defeeb85af99adac4 \
 --duckdb "${D}/tools/duckdb-1.5.4/duckdb" --expected-duckdb-sha256 c478c0f49f6755100d55dee32de485c0f00f3729b40b62658a83b0666b684707 \
 --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 32 --output "${S}/part-${I}.jsonl.gz" --receipt "${S}/part-${I}.receipt.json" --maximum-residues 1020
