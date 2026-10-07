#!/usr/bin/env bash
#SBATCH --job-name=fc-node-cache-canary
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/systems/framecontact_node_cache_v1/logs/canary-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/systems/framecontact_node_cache_v1/logs/canary-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tools=${root}/remote_tools/framecontact_node_cache_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly structure_root=${campaign}/data/structure_cache_plddt70_v1
readonly contact_root=${campaign}/data/contact_cache_plddt70_authority_v1
readonly embedding_root=${root}/data/sppider_seq_sabdab2_embeddings
readonly result_root=${root}/results/systems/framecontact_node_cache_v1
readonly node_cache_root=/tmp/${USER}/framecontact_node_cache_v1
readonly first_receipt=${result_root}/stage-${SLURM_JOB_ID}-first.json
readonly second_receipt=${result_root}/stage-${SLURM_JOB_ID}-reuse.json

test "${SLURM_JOB_PARTITION}" = gpu_4090
mkdir -p "${result_root}/logs" "${node_cache_root}"
chmod 700 "${node_cache_root}"

printf '%s  %s\n' \
  a464e5d9d785da104a5f179925298a899e3c80d698831893335a5981a5fa790c "${tools}/stage_framecontact_cache_v1.py" \
  | sha256sum --check --strict

stage_command=(
  "${python_bin}" "${tools}/stage_framecontact_cache_v1.py"
  --source-root "structure=${structure_root}"
  --manifest-relative "structure=STRUCTURE_CACHE_MANIFEST.json"
  --manifest-sha256 "structure=1c49c5d49742bd7f1a2dba5c52539750cbd90ef03f0b4778492abe0955e496e5"
  --source-root "contact=${contact_root}"
  --manifest-relative "contact=CONTACT_CACHE_MANIFEST.json"
  --manifest-sha256 "contact=cf0c3ca071ed2a7bcd00bfcea519a33bf02a3a7661c1d2dfd2cd146a8b47a12d"
  --source-root "embedding=${embedding_root}"
  --manifest-relative "embedding=SABDAB2_ESM_EMBEDDING_MANIFEST.json"
  --manifest-sha256 "embedding=9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545"
  --node-cache-root "${node_cache_root}"
)

SECONDS=0
"${stage_command[@]}" --durable-receipt "${first_receipt}"
readonly first_elapsed=${SECONDS}
SECONDS=0
"${stage_command[@]}" --durable-receipt "${second_receipt}"
readonly second_elapsed=${SECONDS}

"${python_bin}" - "${first_receipt}" "${second_receipt}" \
  "${first_elapsed}" "${second_elapsed}" "${result_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

first_path, second_path, first_elapsed, second_elapsed, result_root = sys.argv[1:]
first = json.load(open(first_path, encoding="utf-8"))
second = json.load(open(second_path, encoding="utf-8"))
assert first["status"] == "PASS_NODE_LOCAL_CACHE_VERIFIED"
assert second["status"] == "PASS_NODE_LOCAL_CACHE_VERIFIED"
assert first["spec_sha256"] == second["spec_sha256"]
assert first["reused_existing_cache"] is False
assert second["reused_existing_cache"] is True
assert first["scientific_protocol_changed"] is False
assert first["test_or_temporal_access"] is False

def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

result = {
    "first_stage_elapsed_seconds": int(first_elapsed),
    "first_stage_receipt_sha256": sha(first_path),
    "format_version": "framecontact_node_local_cache_canary_v1",
    "node": __import__("socket").gethostname(),
    "reuse_elapsed_seconds": int(second_elapsed),
    "reuse_receipt_sha256": sha(second_path),
    "scientific_protocol_changed": False,
    "spec_sha256": first["spec_sha256"],
    "status": "PASS_NODE_LOCAL_CACHE_CANARY",
    "test_or_temporal_access": False,
}
path = Path(result_root) / f"canary-{__import__('os').environ['SLURM_JOB_ID']}.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY
