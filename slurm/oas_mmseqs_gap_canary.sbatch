#!/bin/bash
#SBATCH --job-name=oas-mmseqs-gap-canary
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-mmseqs-gap-canary-%j.err

set -euo pipefail
umask 077

PYTHON_BIN=${ABF_PYTHON_BIN:-/ssd/scxi253/model_deployment/envs/frontier_torch213/bin/python}
MMSEQS_BIN=${ABF_MMSEQS_BIN:?ABF_MMSEQS_BIN is required}
SOURCE_ROOT=${ABF_SOURCE_ROOT:?ABF_SOURCE_ROOT is required}
SOURCE_COMMIT=${ABF_SOURCE_COMMIT:?ABF_SOURCE_COMMIT is required}
OUTPUT_ROOT=${ABF_OUTPUT_ROOT:?ABF_OUTPUT_ROOT is required}

[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]]
[[ -x "${PYTHON_BIN}" && -x "${MMSEQS_BIN}" ]]
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]]
IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]]
mkdir -p "${OUTPUT_ROOT}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${SOURCE_ROOT}/src
export TMPDIR=${OUTPUT_ROOT}/tmp
export XDG_CACHE_HOME=${OUTPUT_ROOT}/cache
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

CONTROL_ROOT=${OUTPUT_ROOT}/controls
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_mmseqs_gap_canary.py" \
  prepare --output-root "${CONTROL_ROOT}"

QUERY_DB=${OUTPUT_ROOT}/query_db
TARGET_DB=${OUTPUT_ROOT}/target_db
TARGET_GPU=${OUTPUT_ROOT}/target_gpu
PREFILTER=${OUTPUT_ROOT}/prefilter
PREFILTER_CAP1=${OUTPUT_ROOT}/prefilter_cap1
ALIGNMENT_DB=${OUTPUT_ROOT}/alignment_db
ALIGNMENTS=${OUTPUT_ROOT}/alignments.tsv
"${MMSEQS_BIN}" createdb "${CONTROL_ROOT}/queries.fasta" "${QUERY_DB}" --shuffle 0
"${MMSEQS_BIN}" createdb "${CONTROL_ROOT}/targets.fasta" "${TARGET_DB}" --shuffle 0
"${MMSEQS_BIN}" makepaddedseqdb \
  "${TARGET_DB}" "${TARGET_GPU}" --mask 0 --threads 8
"${MMSEQS_BIN}" ungappedprefilter \
  "${QUERY_DB}" "${TARGET_GPU}" "${PREFILTER}" \
  --gpu 1 --prefilter-mode 1 --max-seqs 100 --comp-bias-corr 0 \
  -e 10000 --threads 8
"${MMSEQS_BIN}" align \
  "${QUERY_DB}" "${TARGET_DB}" "${PREFILTER}" "${ALIGNMENT_DB}" \
  -a 1 --alignment-mode 3 --min-seq-id 0.9 -c 0.9 --cov-mode 0 --seq-id-mode 0 \
  --max-accept 2147483647 --max-rejected 2147483647 --threads 8
"${MMSEQS_BIN}" convertalis \
  "${QUERY_DB}" "${TARGET_DB}" "${ALIGNMENT_DB}" "${ALIGNMENTS}" \
  --format-output query,target,fident,qcov,tcov,alnlen,nident,qstart,qend,qlen,tstart,tend,tlen,qaln,taln \
  --threads 8
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_mmseqs_gap_canary.py" \
  audit-positive \
  --expected "${CONTROL_ROOT}/expected.json" \
  --prefilter-prefix "${PREFILTER}" \
  --maximum-sequences 100 \
  --alignments "${ALIGNMENTS}" \
  --output "${OUTPUT_ROOT}/CANARY_RESULT.json"

"${MMSEQS_BIN}" ungappedprefilter \
  "${QUERY_DB}" "${TARGET_GPU}" "${PREFILTER_CAP1}" \
  --gpu 1 --prefilter-mode 1 --max-seqs 1 --comp-bias-corr 0 \
  -e 10000 --threads 8
set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_oas_mmseqs_gap_canary.py" \
  audit-prefilter --prefix "${PREFILTER_CAP1}" --query-count 4 --maximum-sequences 1 \
  > "${OUTPUT_ROOT}/negative.stdout" 2> "${OUTPUT_ROOT}/negative.stderr"
NEGATIVE_STATUS=$?
set -e
[[ "${NEGATIVE_STATUS}" == 3 ]] \
  || { echo "Cap-saturation negative control did not fail closed" >&2; exit 3; }
grep -F "truncation is possible" "${OUTPUT_ROOT}/negative.stderr"
{
  echo "job_id=${SLURM_JOB_ID}"
  echo "node=$(hostname)"
  echo "source_commit=${SOURCE_COMMIT}"
  echo "mmseqs_version=$("${MMSEQS_BIN}" version)"
  echo "negative_control_exit=${NEGATIVE_STATUS}"
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
} > "${OUTPUT_ROOT}/environment.txt"
chmod -R a-w "${OUTPUT_ROOT}"
echo "oas_mmseqs_gap_canary=PASS"
