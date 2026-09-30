#!/bin/bash
# Fail-closed full SaProt/ProSST programme submission.
set -Eeuo pipefail

: "${PROGRAM_ROOT:?PROGRAM_ROOT required}"
: "${ATTEMPT_ROOT:?ATTEMPT_ROOT required}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT required}"

PY=/ssd/scxi253/repr_extract/venv/bin/python3
SOURCE="$PROGRAM_ROOT/source"
SUBMITTED=()

cleanup_on_error() {
  status=$?
  if (( ${#SUBMITTED[@]} )); then
    scancel "${SUBMITTED[@]}" 2>/dev/null || true
  fi
  printf 'submission failed status=%s submitted=%s\n' "$status" "${SUBMITTED[*]:-none}" > "$ATTEMPT_ROOT/SUBMISSION_FAILURE.txt"
  exit "$status"
}
trap cleanup_on_error ERR

test ! -e "$ATTEMPT_ROOT"
mkdir -p "$ATTEMPT_ROOT/logs"
cd "$SOURCE"
export PYTHONPATH="$PROGRAM_ROOT/pylibs:$PWD/src:$PWD"

"$PY" -u extensions/frontier_multimodel_v1/structural_plm_full.py freeze \
  --structure-units "$PROGRAM_ROOT/data/structure_units_v2/STRUCTURE_UNITS_MANIFEST.json" \
  --saprot-tokens "$PROGRAM_ROOT/tokens/saprot_v2" \
  --model-authority "$PROGRAM_ROOT/receipts/MODEL_RUNTIME_AUTHORITY.json" \
  --programme-root "$PROGRAM_ROOT" \
  --output-root "$ATTEMPT_ROOT/output" \
  --commit "$SOURCE_COMMIT" \
  --plan "$ATTEMPT_ROOT/PLAN.json" > "$ATTEMPT_ROOT/PLAN_FREEZE.txt"

test -s "$ATTEMPT_ROOT/PLAN.json"
"$PY" - "$ATTEMPT_ROOT/PLAN.json" "$SOURCE_COMMIT" <<'PLAN_CHECK'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); expected=sys.argv[2]
d=json.loads(p.read_text())
assert d['format_version']=='structural_plm_full_plan_v1'
assert d['commit']==expected
assert d['record_count']==3345
assert d['counts_by_side']=={'antibody':2174,'antigen':1171}
assert d['layers']=={'saprot':[0,8,16,24,33],'prosst':[0,3,6,9,12]}
PLAN_CHECK
PLAN_SHA=$(sha256sum "$ATTEMPT_ROOT/PLAN.json" | cut -d' ' -f1)
[[ "$PLAN_SHA" =~ ^[0-9a-f]{64}$ ]]
printf '%s\n' "$PLAN_SHA" > "$ATTEMPT_ROOT/PLAN.sha256"
sha256sum \
  extensions/frontier_multimodel_v1/structural_plm.py \
  extensions/frontier_multimodel_v1/structural_plm_full.py \
  extensions/frontier_multimodel_v1/saprot_full_extract.sbatch \
  extensions/frontier_multimodel_v1/prosst_full_quant.sbatch \
  extensions/frontier_multimodel_v1/prosst_full_extract.sbatch \
  extensions/frontier_multimodel_v1/structural_plm_full_audit.sbatch \
  > "$ATTEMPT_ROOT/SOURCE_SHA256SUMS.txt"

COMMON="ALL,PROGRAM_ROOT=$PROGRAM_ROOT,ATTEMPT_ROOT=$ATTEMPT_ROOT,PLAN_SHA256=$PLAN_SHA"
sbatch --test-only --array=0-31%8 --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/saprot_full_extract.sbatch" > "$ATTEMPT_ROOT/PREFLIGHT_SAPROT.txt" 2>&1
sbatch --test-only --array=0-63%16 --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_quant.sbatch" > "$ATTEMPT_ROOT/PREFLIGHT_PROSST_QUANT.txt" 2>&1
sbatch --test-only --array=0-31%8 --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_extract.sbatch" > "$ATTEMPT_ROOT/PREFLIGHT_PROSST_EXTRACT.txt" 2>&1
sbatch --test-only --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/structural_plm_full_audit.sbatch" > "$ATTEMPT_ROOT/PREFLIGHT_AUDIT.txt" 2>&1

S=$(sbatch --parsable --array=0-31%8 --export="$COMMON" --output="$ATTEMPT_ROOT/logs/saprot-%A_%a.out" --error="$ATTEMPT_ROOT/logs/saprot-%A_%a.err" "$SOURCE/extensions/frontier_multimodel_v1/saprot_full_extract.sbatch")
SUBMITTED+=("$S")
printf '%s\n' "$S" > "$ATTEMPT_ROOT/SAPROT_ARRAY_JOB_ID.txt"
Q=$(sbatch --parsable --array=0-63%16 --export="$COMMON" --output="$ATTEMPT_ROOT/logs/prosst-quant-%A_%a.out" --error="$ATTEMPT_ROOT/logs/prosst-quant-%A_%a.err" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_quant.sbatch")
SUBMITTED+=("$Q")
printf '%s\n' "$Q" > "$ATTEMPT_ROOT/PROSST_QUANT_ARRAY_JOB_ID.txt"
P=$(sbatch --parsable --dependency="afterok:$Q" --array=0-31%8 --export="$COMMON" --output="$ATTEMPT_ROOT/logs/prosst-%A_%a.out" --error="$ATTEMPT_ROOT/logs/prosst-%A_%a.err" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_extract.sbatch")
SUBMITTED+=("$P")
printf '%s\n' "$P" > "$ATTEMPT_ROOT/PROSST_EXTRACT_ARRAY_JOB_ID.txt"
D=$(sbatch --parsable --dependency="afterok:$S:$P" --export="$COMMON" --output="$ATTEMPT_ROOT/logs/audit-%j.out" --error="$ATTEMPT_ROOT/logs/audit-%j.err" "$SOURCE/extensions/frontier_multimodel_v1/structural_plm_full_audit.sbatch")
SUBMITTED+=("$D")
printf '%s\n' "$D" > "$ATTEMPT_ROOT/AUDIT_JOB_ID.txt"

trap - ERR
printf 'PASS_SUBMITTED plan=%s saprot=%s prosst_quant=%s prosst_extract=%s audit=%s\n' "$PLAN_SHA" "$S" "$Q" "$P" "$D" | tee "$ATTEMPT_ROOT/SUBMISSION_RECEIPT.txt"
