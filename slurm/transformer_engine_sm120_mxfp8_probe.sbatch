#!/bin/bash
#SBATCH --job-name=te218-sm120-mxfp8
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/ssd/scxi253/model_deployment/logs/te218-sm120-mxfp8-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/te218-sm120-mxfp8-%j.err

set -euo pipefail
umask 077

SOURCE_ROOT=${TE_SOURCE_ROOT:?TE_SOURCE_ROOT is required}
SOURCE_COMMIT=${TE_SOURCE_COMMIT:?TE_SOURCE_COMMIT is required}
BASE_PYTHON=${TE_BASE_PYTHON:?TE_BASE_PYTHON is required}
BASE_PYTHON_SHA256=${TE_BASE_PYTHON_SHA256:?required}
BASE_CUDA_ROOT=${TE_BASE_CUDA_ROOT:?TE_BASE_CUDA_ROOT is required}
BASE_CUDNN_ROOT=${TE_BASE_CUDNN_ROOT:?TE_BASE_CUDNN_ROOT is required}
BASE_NCCL_ROOT=${TE_BASE_NCCL_ROOT:?TE_BASE_NCCL_ROOT is required}
OVERLAY_ROOT=${TE_OVERLAY_ROOT:?TE_OVERLAY_ROOT is required}
OVERLAY_FILE_MANIFEST_SHA256=${TE_OVERLAY_FILE_MANIFEST_SHA256:?required}
PREREQUISITE_RECEIPT_SHA256=${TE_PREREQUISITE_RECEIPT_SHA256:?required}
CONTROLLER_SHA256=${TE_CONTROLLER_SHA256:?required}
OUTPUT_ROOT=${TE_OUTPUT_ROOT:?TE_OUTPUT_ROOT is required}
PARTIAL=${OUTPUT_ROOT}.partial-${SLURM_JOB_ID}
CONTROLLER=${SOURCE_ROOT}/scripts/probe_transformer_engine_sm120_mxfp8.py
OVERLAY_MANIFEST=${OVERLAY_ROOT}/file-sha256s.txt
PREREQUISITE_RECEIPT=${OVERLAY_ROOT}/gpu-capability-receipt.json
OUTPUT=${PARTIAL}/gpu-capability-receipt.json

die() {
  echo "Transformer Engine MXFP8 probe: $*" >&2
  exit 2
}

for path in \
  "${SOURCE_ROOT}" "${BASE_PYTHON}" "${BASE_CUDA_ROOT}" \
  "${BASE_CUDNN_ROOT}" "${BASE_NCCL_ROOT}" "${OVERLAY_ROOT}" \
  "${OUTPUT_ROOT}"; do
  [[ "${path}" == /* ]] || die "all paths must be absolute: ${path}"
done
for digest in \
  "${BASE_PYTHON_SHA256}" "${OVERLAY_FILE_MANIFEST_SHA256}" \
  "${PREREQUISITE_RECEIPT_SHA256}" "${CONTROLLER_SHA256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA-256 input"
done
[[ "${SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "source commit is invalid"
[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]] || die "source root differs"
[[ -x "${BASE_PYTHON}" ]] || die "base Python is unavailable"
[[ -d "${OVERLAY_ROOT}" && ! -L "${OVERLAY_ROOT}" ]] || die "overlay is unavailable"
[[ -f "${OVERLAY_MANIFEST}" && ! -L "${OVERLAY_MANIFEST}" ]] \
  || die "overlay file manifest is unavailable"
[[ -f "${PREREQUISITE_RECEIPT}" && ! -L "${PREREQUISITE_RECEIPT}" ]] \
  || die "prerequisite receipt is unavailable"
[[ ! -e "${OUTPUT_ROOT}" && ! -L "${OUTPUT_ROOT}" ]] \
  || die "output root already exists"
[[ ! -e "${PARTIAL}" && ! -L "${PARTIAL}" ]] \
  || die "partial output already exists"
[[ "$(<"${SOURCE_ROOT}/SOURCE_COMMIT")" == "${SOURCE_COMMIT}" ]] \
  || die "source commit differs"
BASE_PYTHON_RESOLVED=$(readlink -f -- "${BASE_PYTHON}")
[[ -f "${BASE_PYTHON_RESOLVED}" && ! -L "${BASE_PYTHON_RESOLVED}" ]] \
  || die "resolved base Python differs"
[[ "$(sha256sum "${BASE_PYTHON_RESOLVED}" | awk '{print $1}')" == "${BASE_PYTHON_SHA256}" ]] \
  || die "base Python SHA-256 differs"
[[ "$(sha256sum "${CONTROLLER}" | awk '{print $1}')" == "${CONTROLLER_SHA256}" ]] \
  || die "controller SHA-256 differs"
[[ "$(sha256sum "${OVERLAY_MANIFEST}" | awk '{print $1}')" == "${OVERLAY_FILE_MANIFEST_SHA256}" ]] \
  || die "overlay file manifest SHA-256 differs"
[[ "$(sha256sum "${PREREQUISITE_RECEIPT}" | awk '{print $1}')" == "${PREREQUISITE_RECEIPT_SHA256}" ]] \
  || die "prerequisite receipt SHA-256 differs"

"${BASE_PYTHON}" - "${OVERLAY_ROOT}" "${OVERLAY_MANIFEST}" <<'PY'
import hashlib
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
seen = 0
for line in manifest.read_text(encoding="ascii").splitlines():
    digest, recorded = line.split("  ", 1)
    marker = "/site-packages/"
    if len(digest) != 64 or marker not in recorded:
        raise SystemExit("invalid overlay manifest entry")
    path = root / "site-packages" / recorded.split(marker, 1)[1]
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit("overlay entry is absent, symbolic, or non-regular")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != digest:
        raise SystemExit("overlay entry SHA-256 differs")
    seen += 1
if seen == 0:
    raise SystemExit("overlay manifest is empty")
print(f"verified_overlay_files={seen}")
PY

IFS=',' read -r -a ALLOCATED_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:?required}"
[[ "${#ALLOCATED_GPU_IDS[@]}" == 1 ]] || die "probe requires one scheduler GPU"
[[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" == 1 ]] \
  || die "probe requires one visible GPU"

mkdir -m 700 "${PARTIAL}"
trap 'chmod -R a-w "${PARTIAL}" 2>/dev/null || true' EXIT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=${OVERLAY_ROOT}/site-packages
export LD_LIBRARY_PATH=${BASE_CUDA_ROOT}/lib:${BASE_CUDNN_ROOT}/lib:${BASE_NCCL_ROOT}/lib
"${BASE_PYTHON}" "${CONTROLLER}" \
  --expected-device-name "NVIDIA GeForce RTX 5090" \
  --expected-compute-capability 12.0 \
  --expected-transformer-engine-version 2.18.0 \
  --prerequisite-receipt "${PREREQUISITE_RECEIPT}" \
  --output "${OUTPUT}"
sha256sum "${OUTPUT}" > "${PARTIAL}/receipt-file-sha256.txt"
chmod -R a-w "${PARTIAL}"
mv "${PARTIAL}" "${OUTPUT_ROOT}"
trap - EXIT
echo "transformer_engine_sm120_mxfp8_probe=PASS"
