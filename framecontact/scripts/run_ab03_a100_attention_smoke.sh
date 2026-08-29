#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 SOURCE_MANIFEST_SHA256 OUTPUT_ROOT PACKAGE_ARCHIVE_SHA256" >&2
  exit 2
fi

source_manifest_sha256="$1"
output_root="$2"
package_archive_sha256="$3"
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_manifest="$source_root/SOURCE_MANIFEST.json"
image="liyin/sabdab2-geometry-a100-replay:sklearn-1.6.1-872701e1"
expected_image_id="sha256:700902df319a2b2ebb8e455f0402cd435c5928582ea9170ba23c89d79116551c"

if [[ ! "$source_manifest_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "SOURCE_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! "$package_archive_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "PACKAGE_ARCHIVE_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 2
fi
if [[ ! -f "$source_manifest" || -L "$source_manifest" ]]; then
  echo "SOURCE_MANIFEST.json must be a regular non-symlink file" >&2
  exit 2
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite output root: $output_root" >&2
  exit 2
fi
observed_image_id="$(docker image inspect "$image" --format '{{.Id}}')"
if [[ "$observed_image_id" != "$expected_image_id" ]]; then
  echo "A100 smoke image ID differs from the frozen value" >&2
  exit 2
fi

mkdir -p "$output_root"

docker run --rm \
  --network none \
  --gpus 'device=0' \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  --volume "$source_root:/workspace:ro" \
  --workdir /workspace \
  --env PYTHONPATH=/workspace/src \
  "$image" \
  /workspace/scripts/freeze_training_source_tree.py verify \
  --source-root /workspace \
  --manifest /workspace/SOURCE_MANIFEST.json \
  --manifest-sha256 "$source_manifest_sha256"

docker run --rm \
  --network none \
  --gpus 'device=0' \
  --user "$(id -u):$(id -g)" \
  --entrypoint python \
  --volume "$source_root:/workspace:ro" \
  --volume "$output_root:/output" \
  --workdir /workspace \
  --env PYTHONPATH=/workspace/src \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$image" \
  /workspace/scripts/benchmark_framecontact_attention_kernels.py \
  --output /output/BENCHMARK.json \
  --warmup 2 \
  --repeats 5 \
  --lengths 220x128,220x512,231x2363

python3 - \
  "$source_manifest" \
  "$source_manifest_sha256" \
  "$package_archive_sha256" \
  "$output_root/BENCHMARK.json" \
  "$output_root/RUN_RECEIPT.json" \
  "$image" \
  "$observed_image_id" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

(
    source_manifest_path,
    source_manifest_sha256,
    package_archive_sha256,
    benchmark_path,
    receipt_path,
    image,
    image_id,
) = sys.argv[1:]


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_manifest = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
payload = {
    "benchmark_sha256": sha256_file(benchmark_path),
    "claim_boundary": (
        "Synthetic A100 kernel correctness, latency, and allocator evidence only; "
        "no biological-quality, frozen-test, external-generalization, custom-kernel, "
        "or SOTA claim."
    ),
    "container_image": image,
    "container_image_id": image_id,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "package_archive_sha256": package_archive_sha256,
    "schema_version": 1,
    "source_manifest_sha256": source_manifest_sha256,
    "source_tree_sha256": source_manifest["tree_sha256"],
}
Path(receipt_path).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

sha256sum \
  "$source_manifest" \
  "$output_root/BENCHMARK.json" \
  "$output_root/RUN_RECEIPT.json"
