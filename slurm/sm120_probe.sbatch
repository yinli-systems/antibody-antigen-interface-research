#!/bin/bash
# Does the cu128 build actually train on an RTX 5090?
#
# The cluster's gpu_5090 partition was unusable: its cards are sm_120 and the
# project's torch 2.4.1+cu121 compiles only up to sm_90, so tasks died at
# .to(device) within a minute. A separate venv at
# /ssd/scxi253/repr_extract/venv_sm120 holds torch 2.11.0+cu128, which lists
# sm_120 among its compiled architectures. The original venv is untouched.
#
# Run this before sending any real work to that partition.
#
# torch 2.4.1+cu121 compiles only to sm_90 and every task on gpu_5090 died at
# .to(device) within a minute. The check that matters is not that CUDA is
# available -- it reports available on a card it cannot run -- but that a real
# training step completes and produces finite numbers on the actual data.
#SBATCH --job-name=sm120-probe
#SBATCH --partition=gpu_5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/asep_bench/logs/%x-%j.out
#SBATCH --error=/ssd/scxi253/asep_bench/logs/%x-%j.err
set -euo pipefail
D=/ssd/scxi253/model_deployment/data
S=/ssd/scxi253/repr_ablation/source
P=/ssd/scxi253/repr_extract/venv_sm120/bin/python
export PYTHONPATH="$S/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $S

echo "=== card and build ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
$P - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("compiled archs:", torch.cuda.get_arch_list())
print("device:", torch.cuda.get_device_name(0))
major, minor = torch.cuda.get_device_capability(0)
print(f"device capability: sm_{major}{minor}")
# A kernel launch, not a capability query: capability is reported correctly even
# for cards the build has no kernels for.
a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = (a @ b).sum().item()
assert c == c, "matmul produced NaN"
print("matmul ok:", round(c, 3))
PY

echo "=== a real training step on real data ==="
$P -u scripts/sabdab2_interaction_first.py train \
  --prepare-dir $D/interaction_first_asep --contact-dir $D/interaction_first_asep/contacts \
  --embedding-dir $D/interaction_first_asep/embeddings \
  --split-manifest /ssd/scxi253/asep_bench/asep_epitope_group_split.json \
  --output-dir /ssd/scxi253/asep_bench/sm120_probe --seed 1 --max-epochs 1 \
  --max-train-instances 40 --max-validation-instances 20 \
  --multitask-balancer measure --device cuda \
  --representation-manifest /ssd/scxi253/asep_bench/asep_manifest.csv \
  --esm-if1-root /ssd/scxi253/asep_bench/repr/esm_if1

echo "=== the numbers have to be finite, not merely present ==="
$P - <<'PY'
import json, math
h = json.load(open("/ssd/scxi253/asep_bench/sm120_probe/history.json"))["history"]
train = h[-1]["train"]
validation = h[-1]["validation"]
for name, block in (("train", train), ("validation", validation)):
    for key, value in block.items():
        assert isinstance(value, (int, float)) and math.isfinite(value), f"{name}.{key} = {value}"
print("total", round(train["total"], 6))
print("cos(node,edge)", round(train["balancer_cosine_node_edge"], 4))
print("validation node mcc", round(validation["node_epitope_mcc"], 4))
PY
echo "[sm120-ok]"
