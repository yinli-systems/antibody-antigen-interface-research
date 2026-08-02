#!/usr/bin/env bash
#SBATCH --job-name=fc-usalign-fold-v3
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/audit_source_v1"
audit_root="${data_root}/structural_audit_v3"
split_root="${data_root}/split_matrix_v3"
export_root="${data_root}/antigen_structures_full_v1"
output_root="${data_root}/usalign_boundary_v3"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
usalign="/ssd/scxi253/model_deployment/tools/usalign-177cc8a2bbd3e2a6e9c5faaaa4ff5dfa1e6048f7/source/USalign"
usalign_lock="${data_root}/source/baselines/USALIGN_TOOL_LOCK.json"
structural_audit="${audit_root}/ANTIGEN_FOLD_STRUCTURAL_AUDIT.json"
mkdir -p "${data_root}/slurm"
exec > >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${data_root}/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

"${python_bin}" "${source_root}/scripts/run_usalign_boundary_spot_checks.py" \
  --structural-audit "${structural_audit}" \
  --structural-audit-sha256 "$(sha256sum "${structural_audit}" | awk '{print $1}')" \
  --split-manifest "${split_root}/antigen_fold.csv" \
  --split-manifest-sha256 fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6 \
  --export-manifest "${export_root}/ANTIGEN_STRUCTURE_EXPORT_MANIFEST.csv" \
  --export-manifest-sha256 4882930043a4fc26319119f56b55a15bd729d23fb22ab04c485ce3ac79ada3d5 \
  --structure-root "${export_root}/structures" \
  --usalign-binary "${usalign}" \
  --usalign-binary-sha256 b9b9003ed9a1bdeec61ffa00e6d75ad82b9d3755b7204a9442d07822e6297b50 \
  --usalign-lock "${usalign_lock}" \
  --usalign-lock-sha256 ae2d262356516b53845a1fec7a82a6d9afca5549cad33962113881422b2e7930 \
  --output-root "${output_root}" \
  --workers 6
sha256sum "${output_root}/USALIGN_BOUNDARY_RECEIPT.json" \
  "${output_root}/USALIGN_BOUNDARY_SPOT_CHECKS.tsv"
