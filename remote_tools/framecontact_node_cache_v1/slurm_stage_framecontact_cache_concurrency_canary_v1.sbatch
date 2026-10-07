#!/usr/bin/env bash
#SBATCH --job-name=fc-node-cache-lock
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:10:00
#SBATCH --output=/ssd/scxi253/model_deployment/results/systems/framecontact_node_cache_v1/logs/concurrency-%j.out
#SBATCH --error=/ssd/scxi253/model_deployment/results/systems/framecontact_node_cache_v1/logs/concurrency-%j.err

set -euo pipefail

readonly root=/ssd/scxi253/model_deployment
readonly tools=${root}/remote_tools/framecontact_node_cache_v1
readonly python_bin=${root}/envs/contact/bin/python3.10
readonly campaign=${root}/results/development/sabdab2_sequence_predicted_geometry_development_v1
readonly structure_root=${campaign}/data/structure_cache_plddt70_v1
readonly contact_root=${campaign}/data/contact_cache_plddt70_authority_v1
readonly embedding_root=${root}/data/sppider_seq_sabdab2_embeddings
readonly result_root=${root}/results/systems/framecontact_node_cache_v1
readonly node_cache_root=/tmp/${USER}/framecontact_node_cache_concurrency_v1/${SLURM_JOB_ID}

test "${SLURM_JOB_PARTITION}" = gpu_4090
test "${node_cache_root}" = "/tmp/${USER}/framecontact_node_cache_concurrency_v1/${SLURM_JOB_ID}"
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

receipts=()
pids=()
SECONDS=0
for worker in 0 1 2 3; do
  receipt=${result_root}/stage-${SLURM_JOB_ID}-concurrent-${worker}.json
  receipts+=("${receipt}")
  "${stage_command[@]}" --durable-receipt "${receipt}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
readonly elapsed=${SECONDS}

"${python_bin}" - "${elapsed}" "${result_root}" "${receipts[@]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

elapsed = int(sys.argv[1])
result_root = Path(sys.argv[2])
receipt_paths = [Path(item) for item in sys.argv[3:]]
if len(receipt_paths) != 4:
    raise SystemExit("expected four concurrent receipts")
receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
if any(item.get("status") != "PASS_NODE_LOCAL_CACHE_VERIFIED" for item in receipts):
    raise SystemExit("a concurrent cache worker did not pass")
if len({item.get("spec_sha256") for item in receipts}) != 1:
    raise SystemExit("concurrent workers resolved different cache specs")
reuse_flags = [bool(item.get("reused_existing_cache")) for item in receipts]
if reuse_flags.count(False) != 1 or reuse_flags.count(True) != 3:
    raise SystemExit(f"unexpected concurrent reuse flags: {reuse_flags}")
if any(item.get("scientific_protocol_changed") is not False for item in receipts):
    raise SystemExit("scientific protocol changed")
if any(item.get("test_or_temporal_access") is not False for item in receipts):
    raise SystemExit("test or temporal access detected")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {
    "concurrent_workers": 4,
    "elapsed_seconds": elapsed,
    "format_version": "framecontact_node_local_cache_concurrency_canary_v1",
    "node": socket.gethostname(),
    "receipt_sha256": {path.name: sha256(path) for path in receipt_paths},
    "reused_existing_cache_count": reuse_flags.count(True),
    "scientific_protocol_changed": False,
    "spec_sha256": receipts[0]["spec_sha256"],
    "status": "PASS_NODE_LOCAL_CACHE_CONCURRENCY_CANARY",
    "test_or_temporal_access": False,
}
path = result_root / f"concurrency-canary-{os.environ['SLURM_JOB_ID']}.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY

if [[ "${node_cache_root}" != "/tmp/${USER}/framecontact_node_cache_concurrency_v1/${SLURM_JOB_ID}" ]]; then
  echo "Refusing unexpected temporary cache root" >&2
  exit 2
fi
rm -rf -- "${node_cache_root}"
