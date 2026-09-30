#!/bin/bash
set -Eeuo pipefail
: "${PROGRAM_ROOT:?}" "${ATTEMPT_ROOT:?}" "${PLAN_SHA256:?}" "${SAPROT_JOB_ID:?}"
SOURCE="$PROGRAM_ROOT/source"
PLAN="$ATTEMPT_ROOT/PLAN.json"
[[ $(sha256sum "$PLAN" | cut -d' ' -f1) == "$PLAN_SHA256" ]]
SUBMITTED=()
cleanup(){ status=$?; ((${#SUBMITTED[@]})) && scancel "${SUBMITTED[@]}" 2>/dev/null || true; printf 'recovery submission failed status=%s jobs=%s\n' "$status" "${SUBMITTED[*]:-none}" > "$ATTEMPT_ROOT/RECOVERY_SUBMISSION_FAILURE.txt"; exit "$status"; }
trap cleanup ERR
COMMON="ALL,PROGRAM_ROOT=$PROGRAM_ROOT,ATTEMPT_ROOT=$ATTEMPT_ROOT,PLAN_SHA256=$PLAN_SHA256"
sbatch --test-only --array=0-63%16 --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_quant.sbatch" > "$ATTEMPT_ROOT/RECOVERY_PREFLIGHT_QUANT.txt" 2>&1
sbatch --test-only --array=0-31%8 --export="$COMMON" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_extract.sbatch" > "$ATTEMPT_ROOT/RECOVERY_PREFLIGHT_EXTRACT.txt" 2>&1
Q=$(sbatch --parsable --array=0-63%16 --export="$COMMON" --output="$ATTEMPT_ROOT/logs/prosst-quant-recovery-%A_%a.out" --error="$ATTEMPT_ROOT/logs/prosst-quant-recovery-%A_%a.err" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_quant.sbatch")
SUBMITTED+=("$Q"); printf '%s\n' "$Q" > "$ATTEMPT_ROOT/PROSST_QUANT_RECOVERY_JOB_ID.txt"
P=$(sbatch --parsable --dependency="afterok:$Q" --array=0-31%8 --export="$COMMON" --output="$ATTEMPT_ROOT/logs/prosst-recovery-%A_%a.out" --error="$ATTEMPT_ROOT/logs/prosst-recovery-%A_%a.err" "$SOURCE/extensions/frontier_multimodel_v1/prosst_full_extract.sbatch")
SUBMITTED+=("$P"); printf '%s\n' "$P" > "$ATTEMPT_ROOT/PROSST_EXTRACT_RECOVERY_JOB_ID.txt"
D=$(sbatch --parsable --dependency="afterok:$SAPROT_JOB_ID:$P" --export="$COMMON" --output="$ATTEMPT_ROOT/logs/audit-recovery-%j.out" --error="$ATTEMPT_ROOT/logs/audit-recovery-%j.err" "$SOURCE/extensions/frontier_multimodel_v1/structural_plm_full_audit.sbatch")
SUBMITTED+=("$D"); printf '%s\n' "$D" > "$ATTEMPT_ROOT/AUDIT_RECOVERY_JOB_ID.txt"
trap - ERR
printf 'PASS_RECOVERY_SUBMITTED quant=%s extract=%s audit=%s\n' "$Q" "$P" "$D" | tee "$ATTEMPT_ROOT/RECOVERY_SUBMISSION_RECEIPT.txt"
