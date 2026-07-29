"""Tests for the frozen AVIDa-hIL6 construct-boundary protocol."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interaction_first.avida_annotation_protocol import (
    AvidaAnnotationProtocolError,
    load_avida_annotation_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "data_registry/annotation_protocols/avida_hil6_v1.json"


def _mutated(tmp_path: Path, mutation) -> Path:
    payload = json.loads(PROTOCOL.read_text())
    mutation(payload)
    output = tmp_path / "protocol.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def test_protocol_loads_with_score_free_pinned_boundaries() -> None:
    payload, digest = load_avida_annotation_protocol(PROTOCOL)
    assert len(digest) == 64
    assert payload["score_free"] is True
    assert payload["vhh_boundary"]["algorithm"]["seq_type_argument"] == "antibody"
    assert payload["vhh_boundary"]["coordinate_semantics"]["query_end"] == (
        "zero_based_inclusive"
    )
    assert payload["antigen_boundary"]["construct"]["mature_chain_start_one_based_inclusive"] == 30


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p.update(score_free=False), "score-free"),
        (
            lambda p: p["vhh_boundary"]["algorithm"].update(seq_type_argument="vhh"),
            "algorithm field",
        ),
        (
            lambda p: p["vhh_boundary"]["coordinate_semantics"].update(
                query_end="zero_based_exclusive"
            ),
            "coordinate semantics",
        ),
        (
            lambda p: p["vhh_boundary"]["acceptance"].update(
                expected_construct_prefix=""
            ),
            "acceptance policy",
        ),
        (
            lambda p: p["vhh_boundary"]["imgt_cdr_ranges_one_based_inclusive"].update(
                cdr3=[105, 116]
            ),
            "CDR ranges",
        ),
        (
            lambda p: p["antigen_boundary"]["authority"].update(accession="P00000"),
            "authority field",
        ),
        (
            lambda p: p["input_binding"].update(expected_antibodies=1),
            "entity counts",
        ),
        (lambda p: p.update(prediction_score=0.9), "fields differ"),
    ],
)
def test_protocol_rejects_semantic_substitution(
    tmp_path: Path, mutation, message: str
) -> None:
    with pytest.raises(AvidaAnnotationProtocolError, match=message):
        load_avida_annotation_protocol(_mutated(tmp_path, mutation))


def test_protocol_rejects_symlink(tmp_path: Path) -> None:
    link = tmp_path / "protocol.json"
    link.symlink_to(PROTOCOL)
    with pytest.raises(AvidaAnnotationProtocolError, match="regular file"):
        load_avida_annotation_protocol(link)
