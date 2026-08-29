from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pplm_frozen_development_retrieval.py"
SPEC = importlib.util.spec_from_file_location("pplm_retrieval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_manifest_pins_only_official_inference_files() -> None:
    assert MODULE.PPLM_REVISION == "1267db4592703095b889cc22da6cf1796ce038b5"
    assert set(MODULE.SOURCE_FILE_SHA256) == {
        "pplm/__init__.py",
        "pplm/constants.py",
        "pplm/data.py",
        "pplm/modules.py",
        "pplm/multihead_attention.py",
        "pplm/pplm.py",
        "pplm/rotary_embedding.py",
        "pplm_ppi/__init__.py",
        "pplm_ppi/model.py",
        "run_pplm-ppi.py",
    }
    assert all(len(value) == 64 for value in MODULE.SOURCE_FILE_SHA256.values())
    assert set(MODULE.IMPLEMENTATION_FILE_SHA256) == {
        "src/antibody_pretraining/pair_aware_attention.py",
        "src/antibody_pretraining/pair_aware_esm2.py",
    }
    assert all(
        len(value) == 64 for value in MODULE.IMPLEMENTATION_FILE_SHA256.values()
    )


def test_reduce_layer_attention_matches_official_pooling_formula() -> None:
    # H=2, B=1, tokens are [CLS,A,A,EOS,CLS,B,B,EOS].
    attention = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    reduced = MODULE.reduce_layer_attention(attention, 4, [(2, 2)])
    row = attention.transpose(0, 1)[0]
    aa = row[:, 1:3, 1:3]
    ab = row[:, 1:3, 5:7]
    ba = row[:, 5:7, 1:3]
    bb = row[:, 5:7, 5:7]
    inter = (ab + ba.transpose(-1, -2)) / 2
    expected = (
        inter.mean((-1, -2)),
        aa.mean((-1, -2)),
        bb.mean((-1, -2)),
        inter.amax((-1, -2)),
        aa.amax((-1, -2)),
        bb.amax((-1, -2)),
    )
    for actual, target in zip(reduced, expected):
        torch.testing.assert_close(actual[0], target)


def test_retrieval_pairs_deduplicate_exact_endpoint_pairs() -> None:
    records = [
        {
            "instance_id": "a",
            "source_antibody_sequence_sha256": "ab1",
            "source_antigen_sequence_sha256": "ag1",
            "antibody_endpoint_sequence": "AAAA",
            "antigen_sequence": "CCCC",
            "wrong_antibody_instance_ids": ["b"],
            "wrong_antigen_instance_ids": ["b"],
        },
        {
            "instance_id": "b",
            "source_antibody_sequence_sha256": "ab2",
            "source_antigen_sequence_sha256": "ag2",
            "antibody_endpoint_sequence": "BBBB",
            "antigen_sequence": "DDDD",
            "wrong_antibody_instance_ids": ["a"],
            "wrong_antigen_instance_ids": ["a"],
        },
    ]
    pairs, index = MODULE.retrieval_pairs(records, records)
    assert set(index) == {"a", "b"}
    assert len(pairs) == 4
    assert pairs[("ordered", "ab1", "ag2")] == ("AAAA", "DDDD")


def test_official_ensemble_averages_member_and_feature_order() -> None:
    class OrderedHead(torch.nn.Module):
        def forward(self, inter, aa, bb, embed_a, embed_b):
            del inter, aa, bb
            return (embed_a[:, :1] + 3 * embed_b[:, :1]).sigmoid()

    batch = 2
    channels = torch.zeros(batch, 660)
    embed_a = torch.tensor([[1.0] * 1280, [2.0] * 1280])
    embed_b = torch.tensor([[3.0] * 1280, [4.0] * 1280])
    features = (
        channels,
        channels,
        channels,
        embed_a,
        embed_b,
        channels,
        channels,
        channels,
        embed_a,
        embed_b,
    )
    heads = torch.nn.ModuleList(OrderedHead() for _ in range(5))
    actual = MODULE.ensemble_probability(features, heads, heads)
    expected = (
        (embed_a[:, 0] + 3 * embed_b[:, 0]).sigmoid()
        + (embed_b[:, 0] + 3 * embed_a[:, 0]).sigmoid()
    ) / 2
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("probability", [1e-5, 0.25, 0.5, 0.75, 1 - 1e-5])
def test_stable_logit_round_trips_probability(probability: float) -> None:
    logit = MODULE.stable_logit(probability)
    assert 1 / (1 + torch.exp(torch.tensor(-logit))).item() == pytest.approx(probability)


def test_full_retrieval_requires_an_exact_accepted_smoke_summary(tmp_path: Path) -> None:
    payload = {
        "authorization": {"development_pair_retrieval": False},
        "equivalence": {
            "batch_padding_vs_singleton_max_abs_delta": 1.6e-6,
            "clean_room_vs_official_probability_max_abs_delta": 3e-8,
            "status": "EXACT_FEATURE_SEMANTICS_ACCEPTED",
            "streaming_vs_official_full_attention_max_abs_delta": 2.5e-6,
        },
        "protocol_sha256": MODULE.ACCEPTED_SMOKE_PROTOCOL_SHA256,
        "status": "SMOKE_NONCLAIM_COMPLETE",
        "test_access": False,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    gate = MODULE.verify_smoke_gate(path, digest)
    assert gate["status"] == "ACCEPTED_BY_VERIFIED_SMOKE_GATE"
    with pytest.raises(ValueError, match="SHA-256 differs"):
        MODULE.verify_smoke_gate(path, "0" * 64)
