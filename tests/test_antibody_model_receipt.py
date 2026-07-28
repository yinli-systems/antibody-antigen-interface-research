import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "results/pretraining/antibody_model_verification_v2.json"


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def test_model_verification_receipt_is_self_anchored_and_exact() -> None:
    payload = RECEIPT_PATH.read_bytes()
    receipt = json.loads(payload.decode("ascii"))
    unsigned = dict(receipt)
    recorded = unsigned.pop("receipt_sha256")
    counts = {
        name: value["total_parameter_count"]
        for name, value in receipt["parameter_receipts"].items()
    }

    assert hashlib.sha256(payload).hexdigest() == (
        "452906c10488d7a05acd12503ccbc4905e60651b2fa579c2420853d40f39ad81"
    )
    assert hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() == recorded
    assert counts == {
        "abf_35m": 31_648_768,
        "abf_300m": 308_365_312,
        "abf_650m": 634_513_920,
    }
    assert all(receipt["checks"].values())
    assert receipt["test_access"] == {
        "downstream_labels_used": False,
        "sabdab2_heldout_accessed": False,
    }
