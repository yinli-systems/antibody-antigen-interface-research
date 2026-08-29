#!/usr/bin/env bash
#SBATCH --job-name=fc-esm-audit-v2
#SBATCH --partition=gpu_4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

data_root="${FRAMECONTACT_DATA_ROOT:-/ssd/scxi253/model_deployment/data/framecontact_20260730}"
source_root="${data_root}/source"
log_root="${data_root}/slurm"
python_bin="/ssd/scxi253/model_deployment/envs/epiformer/bin/python"
sequences_path="${data_root}/authorities/sabdab2_frozen_instances.parquet"
sequences_sha256="9e87fb4241012e44837f700634ef4110c652822e6a1c0c162ea49487f830b107"
prepared_path="${data_root}/authorities/sabdab2_prepared_instances.parquet"
prepared_sha256="a32aa68bb6780d1222b310bab5e4efbf365c9219e08eb8a897379ceded565335"

mkdir -p "${log_root}"
exec > >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out") \
  2> >(tee -a "${log_root}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err" >&2)

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "slurm_job_id=${SLURM_JOB_ID}"
echo "hostname=$(hostname)"
scontrol show job "${SLURM_JOB_ID}" -o

"${python_bin}" - "${sequences_path}" "${sequences_sha256}" \
  "${prepared_path}" "${prepared_sha256}" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, "/ssd/scxi253/model_deployment/data/framecontact_20260730/source/src")
from interaction_first.sabdab2_structure import authenticate_file

authenticate_file(Path(sys.argv[1]), sys.argv[2], "sequence authority preflight")
authenticate_file(Path(sys.argv[3]), sys.argv[4], "prepared authority preflight")
print("authority_preflight=passed")
PY

PYTHONPATH="${source_root}/src" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
"${python_bin}" "${source_root}/scripts/materialize_sabdab2_esm_embeddings.py" \
  --sequences "${sequences_path}" \
  --sequences-sha256 "${sequences_sha256}" \
  --prepared "${prepared_path}" \
  --prepared-sha256 "${prepared_sha256}" \
  --embedding-root /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings \
  --model-root /ssd/scxi253/model_deployment/models/sppider_seq/esm2_t33_650M_UR50D \
  --device cuda \
  --resume

echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sha256sum /ssd/scxi253/model_deployment/data/sppider_seq_sabdab2_embeddings/SABDAB2_ESM_EMBEDDING_MANIFEST.json
