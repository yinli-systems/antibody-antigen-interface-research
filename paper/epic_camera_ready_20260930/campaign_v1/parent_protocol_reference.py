"""Fail-closed rules for the original paper split; no label-dependent splitting."""
from collections import Counter
import hashlib
import json
from pathlib import Path

COUNTS = {"train": 2887, "validation": 321, "test": 455}
TEST_IDS_SHA = "9a42ea95391659b068dd1e1070290368a5f4c440e73a66e0a5ee2577c71efcd8"
SEEDS = (17, 29, 43)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def id_sha(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def validate_split(rows, expected_counts=COUNTS, expected_test_sha=TEST_IDS_SHA):
    ids = [r["instance_id"] for r in rows]
    require(all(isinstance(i, str) and i and i.strip() == i for i in ids), "Invalid instance ID")
    require(len(ids) == len(set(ids)), "Duplicate instance IDs")
    counts = dict(Counter(r["role"] for r in rows))
    require(counts == expected_counts, f"Wrong split counts: {counts}")
    test_ids = [r["instance_id"] for r in rows if r["role"] == "test"]
    require(id_sha(test_ids) == expected_test_sha, "Wrong test membership despite matching count")
    return {"counts": counts, "role_id_sha256": {
        role: id_sha(r["instance_id"] for r in rows if r["role"] == role)
        for role in expected_counts}}


def index_records(records):
    result = {r["instance_id"]: r for r in records}
    require(len(result) == len(records), "Duplicate cache record")
    return result


def safe_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    require(not Path(relative).is_absolute() and root in path.resolve().parents,
            "Artifact path escapes authority root")
    require(not path.is_symlink(), "Symlink artifact is not an immutable input")
    return path


def selection_key(metrics):
    """Retain the baseline's validation epitope-first, contact-tiebreak selection."""
    import math
    key = (metrics["epitope"]["mpc_ap"], metrics["contact"]["mpc_ap"])
    require(all(v is not None and math.isfinite(v) for v in key), "Invalid selection metrics")
    return key


def selection_improves(key, best):
    # Same 1e-4 epitope tolerance as the original baseline trainer.
    return best is None or key[0] > best[0] + 1e-4 or (
        abs(key[0] - best[0]) <= 1e-4 and key[1] > best[1])
