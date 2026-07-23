#!/bin/bash
#SBATCH --job-name=retrieval-controls-finalize
#SBATCH --partition=vip_gpu_5090_USER
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/project/model_deployment/logs/%x-%j.out
#SBATCH --error=/cluster/project/model_deployment/logs/%x-%j.err

set -euo pipefail

DEPLOY_ROOT=/cluster/project/model_deployment
PYTHON_BIN=$DEPLOY_ROOT/envs/epiformer/bin/python
CONTROL_SOURCE_ROOT=$DEPLOY_ROOT/jobs/interaction_first/controls_v1_6_schema_fix_bb574209/sources
SCRIPT=$CONTROL_SOURCE_ROOT/sabdab2_retrieval_controls.py
CONTROL_SOURCE_SHA256=bb5742097c9a0b0f46419b0c3bebd9ccd378c69dc4137483f55d28219ca9a285
CONTROL_PROTOCOL=${RETRIEVAL_CONTROL_PROTOCOL_V1:?Set the physical controls protocol v1 path}
CONTROL_PROTOCOL_SHA256=21c3a0727a1b9d6543a022f6f9d216414ee8f0fc99590b4febca5dcb0d8cdc02
EXECUTION_PROTOCOL=${RETRIEVAL_PROTOCOL_V1_6:?Set the final physical protocol v1.6 path}
EXECUTION_PROTOCOL_SHA256=b5b74452d3483d8b1fdf8c67387e3d38ad2be34a22b65e26a6cab71b0ca73078
CONTROL_BUNDLE=${RETRIEVAL_CONTROL_BUNDLE:?Set the completed train-only control bundle}
CONTROL_BUNDLE_SHA256=${RETRIEVAL_CONTROL_BUNDLE_SHA256:?Set the reviewed bundle SHA-256}
CONCAT_LEDGER=${RETRIEVAL_CONTROL_CONCAT_LEDGER:?Set the completed concat boundary ledger}
CONCAT_LEDGER_SHA256=${RETRIEVAL_CONTROL_CONCAT_LEDGER_SHA256:?Set its exact SHA-256}
CONCAT_VH_VL_LEDGER=${RETRIEVAL_CONTROL_CONCAT_VH_VL_LEDGER:?Set the completed VH/VL boundary ledger}
CONCAT_VH_VL_LEDGER_SHA256=${RETRIEVAL_CONTROL_CONCAT_VH_VL_LEDGER_SHA256:?Set its exact SHA-256}
ARMED=${RETRIEVAL_CONTROL_SUITE_FINALIZATION_ARMED:-NO}

if [[ ! -f "$SCRIPT" || "$(sha256sum "$SCRIPT" | awk '{print $1}')" != "$CONTROL_SOURCE_SHA256" ]]; then
  echo "The frozen control source is missing or changed" >&2
  exit 4
fi
if [[ "$ARMED" != "YES_BOTH_BOUNDARIES_REVIEWED" ]]; then
  echo "Control suite finalization is disarmed until both boundary ledgers are reviewed" >&2
  exit 5
fi

mkdir -p "$DEPLOY_ROOT/logs" "$DEPLOY_ROOT/cache/tmp"
export PYTHONNOUSERSITE=1
export TMPDIR=$DEPLOY_ROOT/cache/tmp
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

"$PYTHON_BIN" "$SCRIPT" finalize-suite \
  --control-bundle "$CONTROL_BUNDLE" \
  --control-bundle-sha256 "$CONTROL_BUNDLE_SHA256" \
  --control-protocol "$CONTROL_PROTOCOL" \
  --control-protocol-sha256 "$CONTROL_PROTOCOL_SHA256" \
  --execution-protocol "$EXECUTION_PROTOCOL" \
  --execution-protocol-sha256 "$EXECUTION_PROTOCOL_SHA256" \
  --concat-ledger "$CONCAT_LEDGER" \
  --concat-ledger-sha256 "$CONCAT_LEDGER_SHA256" \
  --concat-vh-vl-ledger "$CONCAT_VH_VL_LEDGER" \
  --concat-vh-vl-ledger-sha256 "$CONCAT_VH_VL_LEDGER_SHA256"
