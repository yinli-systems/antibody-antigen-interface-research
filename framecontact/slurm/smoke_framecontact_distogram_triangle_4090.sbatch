#!/usr/bin/env bash
#SBATCH --job-name=fc-dist-tri-smoke
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-dist-tri-smoke-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-dist-tri-smoke-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$base/residue-head-source-v1"
embedding_root=/ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings
split_manifest="$base/split_matrix_v3_1/antigen_fold.csv"
receipt="$base/smoke/distogram-triangle-4090-${SLURM_JOB_ID}.json"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$source_root/src"

cd "$source_root"
/ssd/scxi253/model_deployment/envs/contact/bin/python3 \
  scripts/smoke_framecontact_distogram_triangle.py \
  --structure-cache-root "$base/sabdab2/structure_cohort_v2" \
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \
  --contact-cache-root "$base/sabdab2/contact_cohort_v2" \
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \
  --pair-geometry-cache-root "$base/sabdab2/pair-geometry-panel-20260801-v1" \
  --pair-geometry-manifest-sha256 46f02ae1a6aa4262fca2b699d42de52875e637dc4abc1bcd5ba8073f3e06ed03 \
  --embedding-root "$embedding_root" \
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \
  --split-manifest "$split_manifest" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --instance-id 'pdb_00008c3k_B_+' \
  --output "$receipt" \
  --seed 17 \
  --device cuda \
  --precision bf16 \
  --d-model 128 \
  --num-heads 8 \
  --num-self-layers 2

sha256sum "$receipt"
