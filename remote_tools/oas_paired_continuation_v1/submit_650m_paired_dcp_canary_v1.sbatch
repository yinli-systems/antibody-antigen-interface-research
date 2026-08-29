#!/bin/bash
#SBATCH --job-name=oas-paired-650m-dcp-canary-v1
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/dcp-canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/logs/dcp-canary-%j.err

set -euo pipefail

PYTHON=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
SOURCE_ROOT=/ssd/scxi253/model_deployment/oas_paired_continuation_v1_20260825_v3/source
ABF_SOURCE_ROOT=/ssd/scxi253/model_deployment/results/pretraining/antibody-pretraining-rtx4090-5f1f6ac-v1
CAMPAIGN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/antibody_pretraining_rtx4090_30b_5f1f6ac_v1
RUN_ROOT=/ssd/scxi253/model_deployment/results/pretraining/oas_paired_continuation_v1/dcp-canary-v1
RESULT="$RUN_ROOT/OAS_PAIRED_650M_DCP_CANARY_RESULT_V1.json"

test -x "$PYTHON"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
case "$gpu_name" in
  *4090*) ;;
  *) echo "Expected RTX 4090, observed: $gpu_name" >&2; exit 2 ;;
esac

cd "$SOURCE_ROOT"
echo '6007d66b7f2a5722811b6c8fe15ef5f2a7eaa796bca6e689a1360c0a388a1462  audit_650m_paired_dcp_canary.py' | sha256sum -c -
echo 'fcfb69f016681a8d801db213e49f6ff61252f6882d28eb56753cac3dbbdac58a  paired_pretraining.py' | sha256sum -c -
echo '2352a432fd590ccc61f9d43b58f8c023ca2fb6ceef3a93db59c5881d15e63bb7  PROTOCOL_650M_PAIRED_DCP_CANARY_V1.json' | sha256sum -c -

"$PYTHON" audit_650m_paired_dcp_canary.py \
  --abf-source-root "$ABF_SOURCE_ROOT" \
  --checkpoint-root "$CAMPAIGN_ROOT/segment-000035/run/checkpoint" \
  --checkpoint-receipt "$CAMPAIGN_ROOT/segment-000035/run/receipt.json" \
  --model-family "$ABF_SOURCE_ROOT/protocols/antibody_model_family_v1.json" \
  --paired-module "$SOURCE_ROOT/paired_pretraining.py" \
  --protocol "$SOURCE_ROOT/PROTOCOL_650M_PAIRED_DCP_CANARY_V1.json" \
  --output "$RESULT"
