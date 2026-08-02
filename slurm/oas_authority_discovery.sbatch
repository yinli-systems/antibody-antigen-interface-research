#!/bin/bash
#SBATCH --job-name=oas-authority
#SBATCH --partition=vip_gpu_5090_scxi253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --exclude=wqd10nah09g2
#SBATCH --output=/ssd/scxi253/model_deployment/logs/oas-authority-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/logs/oas-authority-%j.err

set -euo pipefail
umask 077

DEPLOY_ROOT=/ssd/scxi253/model_deployment
PYTHON_BIN=${OAS_PYTHON_BIN:-$DEPLOY_ROOT/envs/frontier_torch213/bin/python}
SOURCE_ROOT=${OAS_SOURCE_ROOT:?OAS_SOURCE_ROOT is required}
OUTPUT_ROOT=${OAS_OUTPUT_ROOT:?OAS_OUTPUT_ROOT is required}
SOURCE_CLASS=${OAS_SOURCE_CLASS:-unpaired}
AUTHORITY_SOURCE_SHA256=${OAS_AUTHORITY_SOURCE_SHA256:?OAS_AUTHORITY_SOURCE_SHA256 is required}
CONTROLLER_SOURCE_SHA256=${OAS_CONTROLLER_SOURCE_SHA256:?OAS_CONTROLLER_SOURCE_SHA256 is required}
SCHEDULER_SOURCE_SHA256=${OAS_SCHEDULER_SOURCE_SHA256:?OAS_SCHEDULER_SOURCE_SHA256 is required}

if [[ "$SOURCE_CLASS" != "unpaired" && "$SOURCE_CLASS" != "paired" ]]; then
  echo "OAS_SOURCE_CLASS must be unpaired or paired" >&2
  exit 2
fi
for absolute_path in "$PYTHON_BIN" "$SOURCE_ROOT" "$OUTPUT_ROOT"; do
  if [[ "$absolute_path" != /* ]]; then
    echo "Runtime, source, and output paths must be absolute" >&2
    exit 2
  fi
done
if [[ "$SOURCE_ROOT" == "/" || "$OUTPUT_ROOT" == "/" ]]; then
  echo "Source and output roots cannot be the filesystem root" >&2
  exit 2
fi
case "$OUTPUT_ROOT/" in
  "$SOURCE_ROOT/"*)
    echo "Output root cannot be inside the sealed source tree" >&2
    exit 2
    ;;
esac
for digest in \
  "$AUTHORITY_SOURCE_SHA256" \
  "$CONTROLLER_SOURCE_SHA256" \
  "$SCHEDULER_SOURCE_SHA256"; do
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    echo "All source anchors must be lowercase SHA-256 digests" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "The locked Python runtime is unavailable" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "The authority output root is create-only" >&2
  exit 2
fi

authority_source=$SOURCE_ROOT/src/antibody_pretraining/oas_authority.py
controller_source=$SOURCE_ROOT/scripts/build_oas_authority_manifest.py
if [[ ! -f "$authority_source" || ! -f "$controller_source" ]]; then
  echo "The sealed source tree is incomplete" >&2
  exit 2
fi
if [[ "$(sha256sum "$authority_source" | cut -d' ' -f1)" != "$AUTHORITY_SOURCE_SHA256" ]]; then
  echo "OAS authority implementation digest differs" >&2
  exit 2
fi
if [[ "$(sha256sum "$controller_source" | cut -d' ' -f1)" != "$CONTROLLER_SOURCE_SHA256" ]]; then
  echo "OAS authority controller digest differs" >&2
  exit 2
fi
if [[ "$(sha256sum "$0" | cut -d' ' -f1)" != "$SCHEDULER_SOURCE_SHA256" ]]; then
  echo "OAS authority scheduler digest differs" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mkdir "$OUTPUT_ROOT"
seal_output() {
  local status=$?
  if [[ -d "$OUTPUT_ROOT" && ! -e "$OUTPUT_ROOT/artifact_sha256.txt" ]]; then
    find "$OUTPUT_ROOT" -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -r sha256sum \
      > "$OUTPUT_ROOT/artifact_sha256.txt" 2>/dev/null || true
  fi
  chmod -R a-w "$OUTPUT_ROOT" 2>/dev/null || true
  return "$status"
}
trap seal_output EXIT

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export PYTHONPATH=$SOURCE_ROOT/src

"$PYTHON_BIN" "$controller_source" \
  --source-class "$SOURCE_CLASS" \
  --fetch \
  --raw-response-out "$OUTPUT_ROOT/raw_response.html" \
  --manifest-out "$OUTPUT_ROOT/manifest.json" \
  --receipt-out "$OUTPUT_ROOT/receipt.json" \
  --timeout-seconds 180

