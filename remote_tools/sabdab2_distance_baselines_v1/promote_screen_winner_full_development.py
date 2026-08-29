#!/usr/bin/env python3
"""Freeze and launch the screen winner on full SAbDab2 development data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CANDIDATES = (
    "framecontact_distance_aux",
    "epiformer_style_same_protocol",
    "mipe_style_same_protocol",
)
CODE_HASHES = {
    "scripts/train_framecontact.py": (
        "07ce142ee37c245f9c0d3c8e6705e350d3b758ceaff7385a967756125bbae2fc"
    ),
    "src/interaction_first/framecontact.py": (
        "a85584f288b32e66e6ab34bcd93b0001ffcd2b173bb1b0fc9e3ab48bccd6171c"
    ),
    "src/interaction_first/framecontact_objective.py": (
        "380bca9791f022c5f19238db5fba897ad2350706bb1286b3c1cd80fe4a8eba48"
    ),
}
SOURCE_MANIFEST_SHA256 = (
    "1f776a12869fcf6a32e145bbcc44c579cdec0fe97460ce74a168a8035a4bff68"
)
FULL_SPLIT_SHA256 = (
    "fd1f516839ec9aabc85531ccfa96754887a13d2f54a3867f68bdc8f55d046ce6"
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(path: Path, expected: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA mismatch for {path}: {observed} != {expected}")


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def selected_metrics(result: dict[str, object]) -> tuple[float, float, int]:
    validation = result["selected_checkpoint_validation"]
    selection = result["selection"]
    return (
        float(validation["epitope"]["mean_per_complex_auprc"]),
        float(validation["contact"]["mean_per_complex_auprc"]),
        int(selection["best_epoch"]),
    )


def launcher_text(
    *,
    root: Path,
    source_root: Path,
    authority_root: Path,
    result_root: Path,
    winner: str,
    distance_weight: float,
    contrastive_weight: float,
    protocol_sha: str,
    receipt_sha: str,
    promotion_sha: str,
) -> str:
    full_split = root / "data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv"
    source_manifest = (
        root
        / "data/framecontact_20260730/sabdab_source_distance_baselines_v3_authority"
        / "SABDAB2_DISTANCE_BASELINES_SOURCE_MANIFEST_V3.json"
    )
    protocol = authority_root / "PROTOCOL_FULL_DEVELOPMENT_V1.json"
    receipt = authority_root / "PROTOCOL_FULL_DEVELOPMENT_V1_RECEIPT.json"
    promotion = authority_root / "FULL_DEVELOPMENT_PROMOTION_MANIFEST_V1.json"
    output = result_root / winner / "seed-17"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=fc-full-dev
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output={result_root}/logs/%x-%j.out
#SBATCH --error={result_root}/logs/%x-%j.err

set -euo pipefail

root={root}
source_root={source_root}
data_root=${{root}}/data/framecontact_20260730
python_bin=${{root}}/envs/contact/bin/python3.10
protocol={protocol}
receipt={receipt}
promotion={promotion}
source_manifest={source_manifest}
full_split={full_split}
output={output}

printf '%s  %s\n' \\
  {protocol_sha} "${{protocol}}" \\
  {receipt_sha} "${{receipt}}" \\
  {promotion_sha} "${{promotion}}" \\
  {FULL_SPLIT_SHA256} "${{full_split}}" \\
  {SOURCE_MANIFEST_SHA256} "${{source_manifest}}" \\
  {CODE_HASHES['scripts/train_framecontact.py']} "${{source_root}}/scripts/train_framecontact.py" \\
  {CODE_HASHES['src/interaction_first/framecontact.py']} "${{source_root}}/src/interaction_first/framecontact.py" \\
  {CODE_HASHES['src/interaction_first/framecontact_objective.py']} "${{source_root}}/src/interaction_first/framecontact_objective.py" \\
  | sha256sum --check --strict

"${{python_bin}}" - "${{protocol}}" "${{promotion}}" <<'PY'
import json
import math
import sys

protocol_path, promotion_path = sys.argv[1:]
with open(protocol_path, encoding="utf-8") as handle:
    protocol = json.load(handle)
with open(promotion_path, encoding="utf-8") as handle:
    promotion = json.load(handle)
task = protocol["tasks"]["winner"]
assert promotion["winner"] == "{winner}"
assert task["name"] == "{winner}"
assert task["experiment"] == "{winner}"
assert math.isclose(float(task["distance_loss_weight"]), {distance_weight}, abs_tol=1e-12)
assert math.isclose(float(task["contrastive_loss_weight"]), {contrastive_weight}, abs_tol=1e-12)
assert protocol["claim_boundary"]["test_loaded"] is False
assert promotion["test_loaded"] is False
PY

mkdir -p "$(dirname "${{output}}")" "{result_root}/logs"
cd "${{source_root}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${{source_root}}/src

"${{python_bin}}" scripts/train_framecontact.py \\
  --dataset-backend sabdab2 \\
  --structure-cache-root ${{data_root}}/sabdab2/structure_cohort_v2 \\
  --structure-manifest-sha256 edf74fd704b8b3b4ec9f7c9779e6a548b51e6d18d182642b5323ebc9ea2033f0 \\
  --contact-cache-root ${{data_root}}/sabdab2/contact_cohort_v2 \\
  --contact-manifest-sha256 d91ed42dcc05ee17cc880f7fa6124a7699f11a3beea75aab13f0b9502c274a33 \\
  --embedding-root ${{root}}/data/sppider_seq_sabdab2_embeddings \\
  --embedding-manifest-sha256 9ecbc08815166fb1a2a4a72f86280ffb808a162faf9521838039fb67c05cd545 \\
  --split-manifest ${{full_split}} \\
  --split-manifest-sha256 {FULL_SPLIT_SHA256} \\
  --protocol ${{protocol}} \\
  --protocol-receipt ${{receipt}} \\
  --protocol-receipt-sha256 {receipt_sha} \\
  --source-manifest ${{source_manifest}} \\
  --source-manifest-sha256 {SOURCE_MANIFEST_SHA256} \\
  --training-partner-panel ${{data_root}}/partner_metadata_antigen_fold_v3/train-reciprocal-31.json \\
  --training-partner-panel-sha256 fd64b540ed0788d2e89d9c487d9cbde3cc56c4cb0225f29013007b2c3884efd6 \\
  --validation-partner-panel ${{data_root}}/partner_metadata_antigen_fold_v3/validation-reciprocal-31.json \\
  --validation-partner-panel-sha256 222a22e1c640b679df435e51e00f3550a61a6f880a31efb1807883b520aa30f5 \\
  --output-root ${{output}} \\
  --seed 17 --experiment {winner} \\
  --epochs 20 --early-stopping-patience 5 \\
  --batch-size 1 --evaluation-batch-size 1 --workers 4 \\
  --d-model 128 --num-heads 8 --num-self-layers 2 --num-rbf 16 \\
  --neighbor-k 32 --neighbor-radius 16.0 --sequence-window 32 \\
  --dropout 0.1 --learning-rate 0.0003 --weight-decay 0.0001 \\
  --contact-loss-weight 1.0 --paratope-loss-weight 0.1 \\
  --epitope-loss-weight 0.5 --specificity-loss-weight 0.0 \\
  --distance-loss-weight {distance_weight} \\
  --contrastive-loss-weight {contrastive_weight} \\
  --contrastive-temperature 0.1 --contrastive-maximum-residues 256 \\
  --contrastive-dim 64 \\
  --specificity-margin 0.05 --score-fraction 0.01 \\
  --contact-focal-gamma 1.0 --epitope-dice-weight 1.0 \\
  --epitope-count-weight 0.2 \\
  --calibration-max-pairs 500000 --offload-saved-tensors \\
  --device cuda --precision bf16
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    source_root = (
        root / "data/framecontact_20260730/sabdab_source_distance_baselines_v3"
    )
    source_manifest = (
        root
        / "data/framecontact_20260730/sabdab_source_distance_baselines_v3_authority"
        / "SABDAB2_DISTANCE_BASELINES_SOURCE_MANIFEST_V3.json"
    )
    screen_protocol = source_root / "protocols/protocol_distance_baseline_screen_v3.json"
    full_split = root / "data/framecontact_20260730/split_matrix_v3_1/antigen_fold.csv"
    verify(source_manifest, SOURCE_MANIFEST_SHA256)
    verify(full_split, FULL_SPLIT_SHA256)
    for relative, expected in CODE_HASHES.items():
        verify(source_root / relative, expected)

    split_counts = Counter(
        row["role"]
        for row in csv.DictReader(full_split.open(encoding="utf-8"))
    )
    if split_counts != Counter({"train": 2565, "validation": 549, "test": 549}):
        raise ValueError(f"unexpected full split counts: {split_counts}")

    results: dict[str, dict[str, object]] = {}
    result_hashes: dict[str, str] = {}
    for name in CANDIDATES:
        result_path = (
            arguments.screen_root / name / "seed-17/FRAMECONTACT_DEVELOPMENT_RESULT.json"
        )
        receipt_path = arguments.screen_root / name / "seed-17/RUN_RECEIPT.json"
        result = read_json(result_path)
        receipt = read_json(receipt_path)
        if result["authority"]["test_loaded"] is not False:
            raise ValueError(f"screen result loaded test: {name}")
        if receipt["test_loaded"] is not False:
            raise ValueError(f"screen receipt loaded test: {name}")
        if receipt["result_sha256"] != sha256(result_path):
            raise ValueError(f"screen result receipt mismatch: {name}")
        results[name] = result
        result_hashes[name] = sha256(result_path)

    winner = max(
        CANDIDATES,
        key=lambda name: (
            selected_metrics(results[name])[0],
            selected_metrics(results[name])[1],
            -selected_metrics(results[name])[2],
        ),
    )
    screen = read_json(screen_protocol)
    task = next(
        value for value in screen["tasks"].values() if value["name"] == winner
    )
    protocol = screen
    protocol["schema_version"] = "sabdab2_full_development_promotion_v1"
    protocol["protocol_status"] = "frozen_before_full_development_training"
    protocol["data"].pop("development_screen", None)
    protocol["data"].pop("screen_split_sha256", None)
    protocol["data"]["full_development"] = {
        "split_sha256": FULL_SPLIT_SHA256,
        "train_count": 2565,
        "validation_count": 549,
        "test_count_not_loaded": 549,
        "test_features_or_labels_loaded": False,
    }
    protocol["tasks"] = {"winner": task}
    protocol["claim_boundary"].update(
        {
            "screen_winner": winner,
            "screen_results_sha256": result_hashes,
            "full_development_only": True,
            "test_loaded": False,
            "selection_completed_before_full_development_training": True,
        }
    )
    protocol_bytes = canonical(protocol)
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    receipt = {
        "format_version": "sabdab2_full_development_receipt_v1",
        "protocol_file_sha256": protocol_sha,
        "protocol_canonical_sha256": protocol_sha,
        "test_loaded": False,
        "frozen_before_training": True,
    }
    receipt_bytes = canonical(receipt)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    promotion = {
        "format_version": "sabdab2_screen_winner_promotion_manifest_v1",
        "winner": winner,
        "screen_selection_rule": (
            "validation epitope mean-per-complex AUPRC; contact mean-per-complex "
            "AUPRC; earlier epoch"
        ),
        "screen_results_sha256": result_hashes,
        "screen_selected_metrics": {
            name: {
                "epitope_mean_per_complex_auprc": selected_metrics(result)[0],
                "contact_mean_per_complex_auprc": selected_metrics(result)[1],
                "selected_epoch": selected_metrics(result)[2],
            }
            for name, result in results.items()
        },
        "full_split_sha256": FULL_SPLIT_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "test_loaded": False,
    }
    promotion_bytes = canonical(promotion)
    promotion_sha = hashlib.sha256(promotion_bytes).hexdigest()

    authority_root = arguments.output_root / "authority"
    launcher_root = arguments.output_root / "launchers"
    write_new(authority_root / "PROTOCOL_FULL_DEVELOPMENT_V1.json", protocol_bytes)
    write_new(
        authority_root / "PROTOCOL_FULL_DEVELOPMENT_V1_RECEIPT.json", receipt_bytes
    )
    write_new(
        authority_root / "FULL_DEVELOPMENT_PROMOTION_MANIFEST_V1.json",
        promotion_bytes,
    )
    launcher = launcher_text(
        root=root,
        source_root=source_root,
        authority_root=authority_root,
        result_root=arguments.output_root,
        winner=winner,
        distance_weight=float(task["distance_loss_weight"]),
        contrastive_weight=float(task["contrastive_loss_weight"]),
        protocol_sha=protocol_sha,
        receipt_sha=receipt_sha,
        promotion_sha=promotion_sha,
    )
    launcher_path = launcher_root / "slurm_full_development_winner_v1.sbatch"
    write_new(launcher_path, launcher.encode())
    os.chmod(launcher_path, 0o755)
    print(
        json.dumps(
            {
                "winner": winner,
                "protocol_sha256": protocol_sha,
                "receipt_sha256": receipt_sha,
                "promotion_manifest_sha256": promotion_sha,
                "launcher": str(launcher_path),
                "test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
