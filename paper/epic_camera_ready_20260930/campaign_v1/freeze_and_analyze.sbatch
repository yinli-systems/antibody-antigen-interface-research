#!/usr/bin/env bash
#SBATCH --job-name=epic-rv-freeze
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail

root=${1:?campaign root required}
campaign_protocol=${2:?campaign protocol required}
campaign_sha=${3:?campaign protocol sha required}
analysis_protocol=${4:?analysis protocol required}
analysis_sha=${5:?analysis protocol sha required}
authority_receipt=${6:?authority receipt required}
authority_sha=${7:?authority receipt sha required}
smoke_manifest=${8:?smoke manifest required}
smoke_sha=${9:?smoke manifest sha required}

export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
python=/ssd/scxi253/model_deployment/envs/contact/bin/python3.10
mkdir -p "$root/analysis"

training_manifest="$root/receipts/TRAINING_MANIFEST.json"
"$python" "$root/source/freeze_training.py" \
  --campaign-protocol "$campaign_protocol" \
  --campaign-protocol-sha256 "$campaign_sha" \
  --analysis-protocol "$analysis_protocol" \
  --analysis-protocol-sha256 "$analysis_sha" \
  --authority-receipt "$authority_receipt" \
  --authority-receipt-sha256 "$authority_sha" \
  --smoke-manifest "$smoke_manifest" \
  --smoke-manifest-sha256 "$smoke_sha" \
  --runs-root "$root/runs" \
  --logs-root "$root/logs" \
  --train-job-id-file "$root/receipts/TRAIN_JOB_ID.txt" \
  --output "$training_manifest"
training_sha=$(sha256sum "$training_manifest" | awk '{print $1}')
printf '%s\n' "$training_sha" > "$root/receipts/TRAINING_MANIFEST.sha256"

"$python" "$root/source/aggregate_validation.py" \
  --campaign-protocol "$campaign_protocol" \
  --campaign-protocol-sha256 "$campaign_sha" \
  --analysis-protocol "$analysis_protocol" \
  --analysis-protocol-sha256 "$analysis_sha" \
  --training-manifest "$training_manifest" \
  --training-manifest-sha256 "$training_sha" \
  --output-json "$root/analysis/VALIDATION_ANALYSIS.json" \
  --output-markdown "$root/analysis/VALIDATION_ANALYSIS.md" \
  --receipt "$root/receipts/VALIDATION_ANALYSIS_RECEIPT.json"

"$python" - "$root" "$campaign_sha" "$analysis_sha" "$training_sha" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1])
def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
payload={
    "status":"PASS_POST_TRAINING_FREEZE_AND_VALIDATION_ANALYSIS",
    "campaign_protocol_sha256":sys.argv[2],
    "analysis_protocol_sha256":sys.argv[3],
    "training_manifest_sha256":sys.argv[4],
    "validation_analysis_sha256":sha(root/"analysis/VALIDATION_ANALYSIS.json"),
    "validation_analysis_receipt_sha256":sha(root/"receipts/VALIDATION_ANALYSIS_RECEIPT.json"),
    "test_label_artifacts_opened":False,
    "test_selection_performed":False,
}
out=root/"receipts/POST_TRAINING_COMPLETE.json"
with out.open("x") as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
print("COMPLETE",out,sha(out))
PY
