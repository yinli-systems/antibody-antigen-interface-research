#!/usr/bin/env bash
#SBATCH --job-name=fc-attn-bench
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-attn-bench-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/data/framecontact_20260730/slurm-logs/fc-attn-bench-%j.err

set -euo pipefail

base=/ssd/scxi253/model_deployment/data/framecontact_20260730
source_root="$1"
source_manifest="$2"
source_manifest_sha256="$3"
python_bin=/ssd/scxi253/model_deployment/envs/contact/bin/python3
output_root="$base/smoke/attention-kernel-benchmark-v1-${SLURM_JOB_ID}"

cd "$source_root"
export PYTHONPATH="$source_root/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$python_bin" scripts/freeze_training_source_tree.py verify \
  --source-root "$source_root" \
  --manifest "$source_manifest" \
  --manifest-sha256 "$source_manifest_sha256"

mkdir -p "$output_root"
"$python_bin" scripts/benchmark_framecontact_attention_kernels.py \
  --output "$output_root/BENCHMARK.json" \
  --warmup 3 \
  --repeats 10 \
  --lengths 220x128,220x512,220x1000
sha256sum "$output_root/BENCHMARK.json"
