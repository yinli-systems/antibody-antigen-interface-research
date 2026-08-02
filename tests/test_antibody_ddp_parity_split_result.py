import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = (
    ROOT / "results/pretraining/antibody_ddp_parity_split_v2"
)
PROTOCOL = ROOT / "protocols/antibody_ddp_scientific_parity_gate_v2.json"
MATERIALIZER = ROOT / "scripts/materialize_antibody_ddp_parity_split.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_parity_split_binds_source_and_protocol() -> None:
    receipt = json.loads(
        (SPLIT_ROOT / "receipt.json").read_text(encoding="ascii")
    )
    assert receipt["format_version"] == (
        "antibody_ddp_parity_split_receipt_v2"
    )
    assert receipt["dense_manifest_payload_sha256"] == (
        "0f73ccf7f537d223ca6d94b83f08b40a34c23e557f2a747cc32d798c974450f6"
    )
    assert receipt["source"] == {
        "materializer_sha256": _sha256(MATERIALIZER),
        "protocol_sha256": _sha256(PROTOCOL),
    }
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")
    encoded = (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert recorded == hashlib.sha256(encoded).hexdigest()


def test_real_parity_split_files_are_immutable_and_accounted() -> None:
    receipt = json.loads(
        (SPLIT_ROOT / "receipt.json").read_text(encoding="ascii")
    )
    arrays = {}
    for name, expected in receipt["files"].items():
        path = SPLIT_ROOT / name
        assert _sha256(path) == expected["sha256"]
        array = np.load(path, allow_pickle=False)
        assert list(array.shape) == expected["shape"]
        assert str(array.dtype) == expected["dtype"]
        arrays[name] = array
    training = arrays["training_indices.npy"]
    evaluation = arrays["evaluation_indices.npy"]
    assert len(training) == 7300
    assert len(evaluation) == 812
    assert len(np.intersect1d(training, evaluation)) == 0
    assert np.array_equal(
        np.union1d(training, evaluation),
        np.arange(8112, dtype=np.int64),
    )
    assert receipt["accounting"] == {
        "complete": True,
        "disjoint": True,
        "evaluation_count": 812,
        "record_count": 8112,
        "training_count": 7300,
    }


def test_real_parity_split_preserves_test_boundary() -> None:
    receipt = json.loads(
        (SPLIT_ROOT / "receipt.json").read_text(encoding="ascii")
    )
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
    assert receipt["ranking"]["evaluation_maximum_digest"] < (
        receipt["ranking"]["first_training_digest"]
    )
