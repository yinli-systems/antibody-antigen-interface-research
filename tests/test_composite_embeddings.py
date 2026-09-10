"""Concatenating frozen representations from several encoders.

The seam is the one the training path already calls,
``load(key, device) -> (chunks, starts, length)``, so these tests hold the
contract at that seam: the base encoder's columns come first, each source
follows with its own columns and a presence column, widths are decided by the
manifest rather than by what happens to be on disk, and a missing artifact
becomes zeros with a presence flag of 0 instead of a dropped instance.
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
    RepresentationSource,
    build_sources,
    manifest_chain_keys,
)
from interaction_first.pretrained_representations import (  # noqa: E402
    RepresentationError,
    RepresentationUnit,
    build_artifact,
    write_artifact,
)

BASE_DIM = 4
SOURCE_DIM = 3


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _Base:
    """The store this one wraps, reduced to the method that is actually used."""

    def __init__(self, length: int, value: float = 1.0) -> None:
        self.length = length
        self.value = value

    def load(self, key: str, device):  # noqa: ARG002
        return [torch.full((self.length, BASE_DIM), self.value)], [0], self.length

    def summary(self):
        return {"hits": 0}


def _write(root: Path, sequence: str, value: float, dim: int = SOURCE_DIM) -> str:
    tensor = torch.full((len(sequence), dim), value)
    artifact = build_artifact(
        RepresentationUnit(sequence=sequence),
        tensor,
        model_name="test-encoder",
        model_revision="v0",
    )
    write_artifact(root, artifact)
    return _digest(sequence)


def test_the_base_columns_come_first_and_are_unchanged(tmp_path: Path) -> None:
    key = _write(tmp_path, "AAAA", 7.0)
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4, value=2.0), [source])

    chunks, starts, length = store.load(key, "cpu")

    assert starts == [0] and length == 4
    assert torch.equal(chunks[0][:, :BASE_DIM], torch.full((4, BASE_DIM), 2.0))


def test_a_source_contributes_its_features_then_a_presence_column(tmp_path: Path) -> None:
    key = _write(tmp_path, "AAAA", 7.0)
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4), [source])

    merged = store.load(key, "cpu")[0][0]

    assert merged.shape == (4, BASE_DIM + SOURCE_DIM + PRESENCE_COLUMNS)
    assert torch.equal(merged[:, BASE_DIM : BASE_DIM + SOURCE_DIM], torch.full((4, SOURCE_DIM), 7.0))
    assert torch.equal(merged[:, -1], torch.ones(4))


def test_a_missing_artifact_is_zero_filled_and_flagged(tmp_path: Path) -> None:
    """The failure this exists for: dropping the instance would change the cohort."""
    key = _digest("AAAA")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4), [source])

    merged = store.load(key, "cpu")[0][0]

    assert merged.shape == (4, BASE_DIM + SOURCE_DIM + PRESENCE_COLUMNS)
    assert torch.equal(merged[:, BASE_DIM : BASE_DIM + SOURCE_DIM], torch.zeros(4, SOURCE_DIM))
    assert torch.equal(merged[:, -1], torch.zeros(4))
    assert store.filled["s"] == 1


def test_zero_filling_is_distinguishable_from_a_genuinely_zero_feature(tmp_path: Path) -> None:
    """Without the presence column these two cases are the same tensor."""
    absent = _digest("AAAA")
    present = _write(tmp_path, "CCCC", 0.0)
    source = RepresentationSource(
        "s", tmp_path, SOURCE_DIM, {absent: (absent,), present: (present,)}
    )
    store = CompositeEmbeddingStore(_Base(4), [source])

    zero_feature = store.load(present, "cpu")[0][0]
    missing = store.load(absent, "cpu")[0][0]

    assert torch.equal(zero_feature[:, BASE_DIM:-1], missing[:, BASE_DIM:-1])
    assert not torch.equal(zero_feature[:, -1], missing[:, -1])


def test_parts_are_concatenated_along_residues_in_order(tmp_path: Path) -> None:
    """AntiBERTy encodes heavy and light separately; the join must respect order."""
    heavy = _write(tmp_path, "AAAA", 1.0)
    light = _write(tmp_path, "CC", 2.0)
    combined = _digest("AAAACC")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {combined: (heavy, light)})
    store = CompositeEmbeddingStore(_Base(6), [source])

    features = store.load(combined, "cpu")[0][0][:, BASE_DIM:-1]

    assert torch.equal(features[:4], torch.full((4, SOURCE_DIM), 1.0))
    assert torch.equal(features[4:], torch.full((2, SOURCE_DIM), 2.0))


def test_parts_that_do_not_cover_the_sequence_raise(tmp_path: Path) -> None:
    """A silent length mismatch would misalign every residue after the junction."""
    heavy = _write(tmp_path, "AAAA", 1.0)
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {heavy: (heavy,)})
    store = CompositeEmbeddingStore(_Base(6), [source])

    with pytest.raises(RepresentationError, match="covers 4 residues"):
        store.load(heavy, "cpu")


def test_a_key_the_source_does_not_cover_contributes_no_columns(tmp_path: Path) -> None:
    """AntiBERTy has nothing to say about an antigen, so the sides differ in width."""
    antibody = _write(tmp_path, "AAAA", 1.0)
    antigen = _digest("GGGG")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {antibody: (antibody,)})
    store = CompositeEmbeddingStore(_Base(4), [source])

    assert store.load(antibody, "cpu")[0][0].shape[1] == BASE_DIM + SOURCE_DIM + PRESENCE_COLUMNS
    assert store.load(antigen, "cpu")[0][0].shape[1] == BASE_DIM
    assert store.width_for(antigen, BASE_DIM) == BASE_DIM


def test_sources_are_concatenated_in_the_order_given(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    key = _write(first, "AAAA", 1.0)
    _write(second, "AAAA", 5.0)
    store = CompositeEmbeddingStore(
        _Base(4),
        [
            RepresentationSource("one", first, SOURCE_DIM, {key: (key,)}),
            RepresentationSource("two", second, SOURCE_DIM, {key: (key,)}),
        ],
    )

    merged = store.load(key, "cpu")[0][0]

    assert torch.equal(merged[:, BASE_DIM], torch.ones(4))
    assert torch.equal(merged[:, BASE_DIM + SOURCE_DIM + 1], torch.full((4,), 5.0))


def test_duplicate_source_names_are_rejected(tmp_path: Path) -> None:
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {})
    with pytest.raises(RepresentationError, match="duplicate source"):
        CompositeEmbeddingStore(_Base(4), [source, source])


def test_the_summary_reports_how_often_a_source_was_absent(tmp_path: Path) -> None:
    key = _digest("AAAA")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4), [source])
    store.load(key, "cpu")
    store.load(key, "cpu")

    summary = store.summary()

    assert summary["sources"][0]["loads_zero_filled"] == 2
    assert summary["sources"][0]["keys_covered"] == 1


def _manifest(path: Path) -> Path:
    rows = [
        {"INSTANCE": "paired", "vh_seq": "AAAA", "vl_seq": "CC", "ag_seq": "GGGG"},
        {"INSTANCE": "single_domain", "vh_seq": "TTTT", "vl_seq": "", "ag_seq": "GGGG"},
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_the_antibody_key_matches_the_models_own_convention(tmp_path: Path) -> None:
    """ab_sequence_sha256 is sha256(vh + vl); anything else silently misses."""
    chains = manifest_chain_keys(_manifest(tmp_path / "m.csv"))

    assert chains["paired"]["antibody_key"] == _digest("AAAACC")
    assert chains["single_domain"]["antibody_key"] == _digest("TTTT")


def test_a_single_domain_antibody_has_one_part_not_an_empty_light_chain(tmp_path: Path) -> None:
    """A quarter of this cohort has no light chain; hashing '' would be a real key."""
    chains = manifest_chain_keys(_manifest(tmp_path / "m.csv"))

    assert chains["paired"]["antibody_parts"] == (_digest("AAAA"), _digest("CC"))
    assert chains["single_domain"]["antibody_parts"] == (_digest("TTTT"),)


def test_built_sources_cover_the_sides_each_encoder_actually_saw(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "m.csv")

    antiberty, esm_if1 = build_sources(
        manifest, antiberty_root=tmp_path / "ab", esm_if1_root=tmp_path / "if1"
    )

    assert antiberty.covers(_digest("AAAACC"))
    assert not antiberty.covers(_digest("GGGG"))
    assert esm_if1.covers(_digest("AAAACC")) and esm_if1.covers(_digest("GGGG"))


def test_coverage_is_decided_by_the_manifest_not_by_what_is_on_disk(tmp_path: Path) -> None:
    """Otherwise a side's width would depend on which artifacts happen to exist."""
    manifest = _manifest(tmp_path / "m.csv")

    (antiberty,) = build_sources(manifest, antiberty_root=tmp_path / "absent")

    assert antiberty.covers(_digest("AAAACC"))
    assert antiberty.width == 512 + PRESENCE_COLUMNS


def test_describe_carries_no_count_that_has_not_been_measured(tmp_path: Path) -> None:
    """The defect this exists for: a receipt written before any load reported 0.

    The training config is built before the first batch is read, so a zero-fill
    counter taken there is always zero -- it looked like "no instance was ever
    missing a feature" when the truth was "nothing has been read yet".
    """
    key = _digest("AAAA")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4), [source])

    described = store.describe()

    assert "loads_zero_filled" not in described["sources"][0]
    assert described["sources"][0]["keys_covered"] == 1


def test_summary_reports_the_counter_once_loads_have_happened(tmp_path: Path) -> None:
    key = _digest("AAAA")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    store = CompositeEmbeddingStore(_Base(4), [source])

    before = store.summary()["sources"][0]["loads_zero_filled"]
    store.load(key, "cpu")
    after = store.summary()["sources"][0]["loads_zero_filled"]

    assert (before, after) == (0, 1)


def _write_varying(root: Path, sequence: str, dim: int = SOURCE_DIM) -> str:
    """Distinct rows, so a permutation is visible at all."""
    tensor = torch.arange(len(sequence), dtype=torch.float32).unsqueeze(1).repeat(1, dim)
    artifact = build_artifact(
        RepresentationUnit(sequence=sequence),
        tensor,
        model_name="test-encoder",
        model_revision="v0",
    )
    write_artifact(root, artifact)
    return _digest(sequence)


def test_shuffling_keeps_every_vector_and_changes_only_their_order(tmp_path: Path) -> None:
    """The attribution control: the complex, length and distribution all survive."""
    key = _write_varying(tmp_path, "ACDEFGHIKL")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    plain = CompositeEmbeddingStore(_Base(10), [source])
    shuffled = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=7)

    a = plain.load(key, "cpu")[0][0][:, BASE_DIM:-1]
    b = shuffled.load(key, "cpu")[0][0][:, BASE_DIM:-1]

    assert a.shape == b.shape
    assert torch.equal(a.sum(dim=0), b.sum(dim=0))
    assert torch.equal(a[:, 0].sort().values, b[:, 0].sort().values)
    assert not torch.equal(a, b)


def test_the_permutation_is_the_same_in_every_process(tmp_path: Path) -> None:
    """A fresh permutation per call would be noise injection, a different experiment."""
    key = _write_varying(tmp_path, "ACDEFGHIKL")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})

    one = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=7)
    two = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=7)

    assert torch.equal(one.load(key, "cpu")[0][0], two.load(key, "cpu")[0][0])
    assert torch.equal(one.load(key, "cpu")[0][0], one.load(key, "cpu")[0][0])


def test_a_different_seed_permutes_differently(tmp_path: Path) -> None:
    key = _write_varying(tmp_path, "ACDEFGHIKL")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})

    one = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=7)
    two = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=8)

    assert not torch.equal(one.load(key, "cpu")[0][0], two.load(key, "cpu")[0][0])


def test_shuffling_leaves_the_base_columns_alone(tmp_path: Path) -> None:
    """Only the extra representation is the subject of the control."""
    key = _write_varying(tmp_path, "ACDEFGHIKL")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})
    base = _Base(10, value=3.0)

    merged = CompositeEmbeddingStore(base, [source], shuffle_seed=7).load(key, "cpu")[0][0]

    assert torch.equal(merged[:, :BASE_DIM], torch.full((10, BASE_DIM), 3.0))


def test_the_receipt_says_the_rows_were_shuffled(tmp_path: Path) -> None:
    """A reader must not mistake this arm for one carrying aligned features."""
    key = _write_varying(tmp_path, "ACDEFGHIKL")
    source = RepresentationSource("s", tmp_path, SOURCE_DIM, {key: (key,)})

    described = CompositeEmbeddingStore(_Base(10), [source], shuffle_seed=7).describe()

    assert described["residue_shuffle_seed"] == 7
    assert "correspondence is destroyed" in described["residue_shuffle"]
