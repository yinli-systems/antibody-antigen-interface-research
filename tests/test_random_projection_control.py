"""The width-matched control for the encoder arm.

The encoder arm is wider than the ESM-2 arm and therefore carries 26.2% more
parameters (763,401 -> 963,471), all of them in the input projections that
consume the wider input. A gain measured against the narrow arm cannot separate
"these encoders carry signal" from "a wider input projection helps". This
control appends a fixed random linear projection of the base embedding instead:
identical widths, identical parameter count, and nothing the model could not
already compute from its own input.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from interaction_first.composite_embeddings import (  # noqa: E402
    PRESENCE_COLUMNS,
    CompositeEmbeddingStore,
    RandomProjectionSource,
    build_random_control,
    build_sources,
)

BASE_DIM = 8
DIM = 3


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _Base:
    def __init__(self, length: int) -> None:
        self.length = length
        generator = torch.Generator().manual_seed(7)
        self.tensor = torch.randn(length, BASE_DIM, generator=generator)

    def load(self, key: str, device):  # noqa: ARG002
        return [self.tensor.clone()], [0], self.length


def _manifest(path: Path) -> Path:
    rows = [
        {"INSTANCE": "i1", "vh_seq": "AAAA", "vl_seq": "CC", "ag_seq": "GGGG"},
        {"INSTANCE": "i2", "vh_seq": "TTTT", "vl_seq": "", "ag_seq": "GGGGG"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_projection_is_the_same_every_call(tmp_path: Path) -> None:
    """Otherwise the control would be a different model on every batch."""
    key = _digest("AAAACC")
    source = RandomProjectionSource("r", DIM, 11, {key: ()})
    store = CompositeEmbeddingStore(_Base(6), [source])

    first = store.load(key, "cpu")[0][0]
    second = store.load(key, "cpu")[0][0]

    assert torch.equal(first, second)


def test_two_stores_built_from_the_same_seed_agree(tmp_path: Path) -> None:
    """A run must be reproducible across processes, as the encoder arms are."""
    key = _digest("AAAACC")
    base = _Base(6)
    one = CompositeEmbeddingStore(base, [RandomProjectionSource("r", DIM, 11, {key: ()})])
    two = CompositeEmbeddingStore(base, [RandomProjectionSource("r", DIM, 11, {key: ()})])

    assert torch.equal(one.load(key, "cpu")[0][0], two.load(key, "cpu")[0][0])


def test_a_different_seed_gives_different_columns(tmp_path: Path) -> None:
    key = _digest("AAAACC")
    base = _Base(6)
    one = CompositeEmbeddingStore(base, [RandomProjectionSource("r", DIM, 11, {key: ()})])
    two = CompositeEmbeddingStore(base, [RandomProjectionSource("r", DIM, 12, {key: ()})])

    assert not torch.equal(one.load(key, "cpu")[0][0], two.load(key, "cpu")[0][0])


def test_the_appended_columns_are_a_function_of_the_base(tmp_path: Path) -> None:
    """The control must carry no information the model did not already have."""
    key = _digest("AAAACC")
    base = _Base(6)
    source = RandomProjectionSource("r", DIM, 11, {key: ()})
    store = CompositeEmbeddingStore(base, [source])

    merged = store.load(key, "cpu")[0][0]
    expected = base.tensor @ source._projection(BASE_DIM, "cpu", base.tensor.dtype)

    assert torch.allclose(merged[:, BASE_DIM:-1], expected)


def test_the_presence_column_is_always_set(tmp_path: Path) -> None:
    """A projection is never absent, so the control never zero-fills."""
    key = _digest("AAAACC")
    store = CompositeEmbeddingStore(_Base(6), [RandomProjectionSource("r", DIM, 11, {key: ()})])

    merged = store.load(key, "cpu")[0][0]

    assert torch.equal(merged[:, -1], torch.ones(6))
    assert store.filled["r"] == 0


def test_the_control_matches_the_encoder_arm_width_exactly(tmp_path: Path) -> None:
    """The whole point: same widths on both sides, so same parameter count."""
    manifest = _manifest(tmp_path / "m.csv")
    encoders = build_sources(
        manifest, antiberty_root=tmp_path / "ab", esm_if1_root=tmp_path / "if1"
    )
    control = build_random_control(manifest, dim=512)
    antibody = _digest("AAAACC")
    antigen = _digest("GGGG")

    encoder_store = CompositeEmbeddingStore(_Base(6), encoders)
    control_store = CompositeEmbeddingStore(_Base(6), control)

    for key in (antibody, antigen):
        assert encoder_store.width_for(key, 1280) == control_store.width_for(key, 1280)


def test_the_control_reproduces_the_asymmetry_between_the_sides(tmp_path: Path) -> None:
    """AntiBERTy covers antibodies only, so the control must too."""
    control_store = CompositeEmbeddingStore(
        _Base(6), build_random_control(_manifest(tmp_path / "m.csv"), dim=512)
    )

    antibody = control_store.width_for(_digest("AAAACC"), 1280)
    antigen = control_store.width_for(_digest("GGGG"), 1280)

    assert antibody == 1280 + 2 * (512 + PRESENCE_COLUMNS)
    assert antigen == 1280 + (512 + PRESENCE_COLUMNS)


def test_the_receipt_says_the_columns_are_random(tmp_path: Path) -> None:
    """A reader must not mistake this arm for one carrying real encoders."""
    store = CompositeEmbeddingStore(
        _Base(6), build_random_control(_manifest(tmp_path / "m.csv"))
    )

    described = store.describe()

    assert {entry["kind"] for entry in described["sources"]} == {"random_projection_of_base"}
    assert all("seed" in entry for entry in described["sources"])


def test_a_length_mismatch_raises(tmp_path: Path) -> None:
    key = _digest("AAAACC")
    source = RandomProjectionSource("r", DIM, 11, {key: ()})

    with pytest.raises(Exception, match="projected 6 residues"):
        source.features(key, 99, "cpu", _Base(6).tensor)
