"""Tests for the frozen AVIDa Tier-1 normalization protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_first.avida_normalization_protocol import (
    AvidaNormalizationProtocolError,
    load_avida_normalization_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "data_registry/normalization_protocols/avida_hil6_tier1_v1.json"
)


def test_protocol_freezes_tier1_admission_and_pair_accounting() -> None:
    payload, digest = load_avida_normalization_protocol(PROTOCOL)
    assert digest == "180b740d73eb9485d089e450d7565c23e894e0a7c5e3444bcb74836baadb9161"
    assert payload["admission"]["vhh"]["expected_accepted"] == 37_200
    assert payload["admission"]["vhh"]["expected_quarantined"] == 1_399
    assert payload["admission"]["pair"]["expected_admitted_rows"] == 555_941
    assert payload["admission"]["pair"]["expected_quarantined_rows"] == 17_950
    assert payload["status"] == "frozen_tier1_only"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["admission"]["vhh"].update(expected_accepted=38_599),
        lambda value: value["admission"]["pair"].update(
            expected_quarantined_rows=0
        ),
        lambda value: value.update(license_route="commercial_ok"),
        lambda value: value["prohibitions"].pop(),
    ],
)
def test_protocol_rejects_semantic_mutation(
    tmp_path: Path,
    mutation: object,
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    mutation(payload)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AvidaNormalizationProtocolError):
        load_avida_normalization_protocol(path)


def test_protocol_rejects_symlink(tmp_path: Path) -> None:
    link = tmp_path / "protocol.json"
    link.symlink_to(PROTOCOL)
    with pytest.raises(AvidaNormalizationProtocolError, match="regular file"):
        load_avida_normalization_protocol(link)
