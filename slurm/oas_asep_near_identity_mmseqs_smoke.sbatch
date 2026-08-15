#!/bin/bash
#SBATCH --job-name=oas-mmseqs-smoke
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/oas-mmseqs-smoke-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/oas-mmseqs-smoke-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
MMSEQS_BIN=${ABF_MMSEQS_BIN:?ABF_MMSEQS_BIN is required}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
SOURCE_COMMIT=${ABF_SOURCE_COMMIT:?ABF_SOURCE_COMMIT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}
PROTOCOL_SHA256=${ABF_NEAR_IDENTITY_PROTOCOL_SHA256:?required}

for artifact_path in "${PYTHON_BIN}" "${MMSEQS_BIN}" "${SOURCE_ROOT}" "${OUTPUT_ROOT}"; do
  [[ "${artifact_path}" == /* ]] || { echo "Smoke paths must be absolute" >&2; exit 2; }
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Source commit is invalid" >&2; exit 2; }
[[ "${PROTOCOL_SHA256}" =~ ^[0-9a-f]{64}$ ]] || { echo "Protocol digest is invalid" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -x "${MMSEQS_BIN}" ]]
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]]
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] \
  || { echo "Smoke output already exists" >&2; exit 2; }
SOURCE_COMMIT_ANCHOR=${SOURCE_ROOT}/SOURCE_COMMIT
[[ -f "${SOURCE_COMMIT_ANCHOR}" && ! -L "${SOURCE_COMMIT_ANCHOR}" ]]
[[ "$(<"${SOURCE_COMMIT_ANCHOR}")" == "${SOURCE_COMMIT}" ]]
PROTOCOL=${SOURCE_ROOT}/protocols/oas_asep_near_identity_decontamination_v1.json
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${PROTOCOL_SHA256}" ]]

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] \
  || { echo "MMseqs smoke requires exactly one scheduler-reserved GPU" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT_ROOT}")"
mkdir "${OUTPUT_ROOT}"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "scheduler_reserved_gpu_count=1"
  echo "gpu_compute_permitted=true"
  echo "development_data_accessed=false"
  echo "official_test_accessed=false"
  date --iso-8601=seconds
  "${MMSEQS_BIN}" version
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
    --format=csv
} > "${OUTPUT_ROOT}/environment.txt"

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
CONTROL_ROOT=${OUTPUT_ROOT}/controls
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_near_identity_mmseqs_smoke.py" \
  prepare --output-root "${CONTROL_ROOT}"

TARGET_DB=${OUTPUT_ROOT}/target_db
PADDED_DB=${OUTPUT_ROOT}/target_gpu
TMP_ROOT=${OUTPUT_ROOT}/tmp
HITS=${OUTPUT_ROOT}/hits.tsv
"${MMSEQS_BIN}" createdb "${CONTROL_ROOT}/targets.fasta" "${TARGET_DB}"
"${MMSEQS_BIN}" makepaddedseqdb "${TARGET_DB}" "${PADDED_DB}"
"${MMSEQS_BIN}" easy-search \
  "${CONTROL_ROOT}/queries.fasta" "${PADDED_DB}" "${HITS}" "${TMP_ROOT}" \
  --gpu 1 \
  --min-seq-id 0.9 \
  -c 0.9 \
  --cov-mode 0 \
  --seq-id-mode 0 \
  --max-seqs 100 \
  --format-output query,target,fident,qcov,tcov

RESULT=${OUTPUT_ROOT}/MMSEQS_GPU_SENSITIVITY_RESULT.json
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_near_identity_mmseqs_smoke.py" \
  audit \
  --expected "${CONTROL_ROOT}/expected.json" \
  --hits "${HITS}" \
  --output "${RESULT}"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv \
  >> "${OUTPUT_ROOT}/environment.txt"
date --iso-8601=seconds >> "${OUTPUT_ROOT}/environment.txt"
sha256sum "${RESULT}" > "${OUTPUT_ROOT}/result.sha256"
chmod -R a-w "${OUTPUT_ROOT}"
echo "oas_near_identity_mmseqs_gpu_smoke=PASS"
echo "result=${RESULT}"
