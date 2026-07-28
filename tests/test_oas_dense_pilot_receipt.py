import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/pretraining/oas_dense_pilot_v1_summary.json"


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


def test_real_oas_dense_pilot_summary_is_self_anchored() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="ascii"))
    unsigned = dict(summary)
    recorded = unsigned.pop("summary_sha256")

    assert summary["accepted_unique_records"] == 8_112
    assert summary["source_rows"] == 8_399
    assert summary["claim_boundary"] == {
        "global_production_dedup_complete": False,
        "pilot_only": True,
        "training_started": False,
    }
    assert hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest() == recorded
