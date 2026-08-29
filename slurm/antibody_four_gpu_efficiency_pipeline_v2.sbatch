#!/bin/bash
#SBATCH --job-name=abf-oas-4g-eff-v2
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-eff-v2-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/abf-oas-4g-eff-v2-%j.err

set -euo pipefail

SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
for name in ABF_SHAPE_PROBE_AUDIT ABF_SHAPE_PROBE_AUDIT_FILE_SHA256 \
  ABF_SHAPE_PROBE_AUDIT_PAYLOAD_SHA256 ABF_SHAPE_PROBE_PROTOCOL_FILE_SHA256; do
  [[ -n "${!name:-}" ]] || { echo "${name} is required" >&2; exit 2; }
done

export ABF_EFFICIENCY_GATE_VERSION=v2
export ABF_COMPILED_LENGTH_BUCKET_BOUNDARIES=194
exec bash "${SOURCE_ROOT}/slurm/antibody_four_gpu_efficiency_pipeline.sbatch"
