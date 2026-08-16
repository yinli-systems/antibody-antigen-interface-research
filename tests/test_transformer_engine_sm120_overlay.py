"""Test the non-authorizing RTX 5090 Transformer Engine overlay gate."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/probe_transformer_engine_sm120.py"
SLURM = ROOT / "slurm/transformer_engine_sm120_overlay.sbatch"


def test_probe_is_ascii_syntax_valid_and_fail_closed() -> None:
    source = PROBE.read_text(encoding="ascii")
    ast.parse(source)
    assert "Float8CurrentScaling" in source
    assert "Exactly one CUDA device must be available" in source
    assert "parameter_gradients_finite" in source
    assert "Capability receipt output must be create-only" in source
    assert '"four_gpu_training": False' in source
    assert '"production_pretraining": False' in source
    assert '"scientific_claim": False' in source
    assert "not a numerical-parity, throughput, MFU" in source


def test_overlay_job_is_bash_valid_source_bound_and_atomic() -> None:
    subprocess.run(["bash", "-n", str(SLURM)], check=True)
    source = SLURM.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --cpus-per-task=8" in source
    assert "NVTE_BUILD_USE_NVIDIA_WHEELS=1" in source
    assert "NVTE_PYTORCH_FORCE_BUILD=TRUE" in source
    assert "transformer-engine==${TE_VERSION}" in source
    assert "transformer-engine-cu13==${TE_VERSION}" in source
    assert "transformer-engine-torch==${TE_VERSION}" in source
    assert "nvidia-cuda-nvcc==${NVCC_VERSION}" in source
    assert "--no-build-isolation" in source
    assert "--no-deps" in source
    assert "SOURCE_COMMIT" in source
    assert "BASE_PYTHON_SHA256" in source
    assert "BASE_CUDA_ROOT" in source
    assert "BASE_CUDA_SHA256SUMS" in source
    assert "BASE_CUDNN_ROOT" in source
    assert "BASE_CUDNN_SHA256SUMS" in source
    assert "BASE_NCCL_ROOT" in source
    assert "BASE_NCCL_SHA256SUMS" in source
    assert "CONTROLLER_SHA256" in source
    assert "WHEELHOUSE_MANIFEST_SHA256" in source
    assert "WHEELHOUSE_ENTRY_COUNT" in source
    assert "wheelhouse inventory differs" in source
    assert "PIP_OFFLINE_ARGS=(--no-index --find-links" in source
    assert 'sha256sum "${WHEELHOUSE}/SHA256SUMS"' in source
    assert 'export CUDA_HOME=${PARTIAL}/site-packages/nvidia/cu13' in source
    assert '[[ -x "${CUDA_HOME}/bin/nvcc"' in source
    assert '[[ "${nvcc_version}" == "13.0" ]]' in source
    assert "nvidia-cuda-cccl==${CUDA_CCCL_VERSION}" in source
    assert "sha256sum --quiet -c" in source
    assert "${BASE_CUDNN_ROOT}/include:${BASE_NCCL_ROOT}/include" in source
    assert "${BASE_CUDNN_ROOT}/lib:${BASE_NCCL_ROOT}/lib" in source
    assert "export LD_LIBRARY_PATH=${BASE_CUDA_ROOT}/lib" in source
    assert "nvidia-cuda-crt==${CUDA_CRT_VERSION}" in source
    assert "nvidia-cuda-runtime==${CUDA_RUNTIME_VERSION}" in source
    assert "NVTE_WITH_NCCL_EP=0" in source
    assert 'BASE_PYTHON_RESOLVED=$(readlink -f -- "${BASE_PYTHON}")' in source
    assert "base-python-invocation-path.txt" in source
    assert "base-python-resolved-path.txt" in source
    assert '[[ -x "${BASE_PYTHON}" ]]' in source
    assert '[[ -x "${BASE_PYTHON}" && ! -L "${BASE_PYTHON}" ]]' not in source
    assert "Transformer Engine overlay gate:" in source
    assert '[[ ! -e "${OUTPUT_ROOT}"' in source
    assert 'mv "${PARTIAL}" "${OUTPUT_ROOT}"' in source
