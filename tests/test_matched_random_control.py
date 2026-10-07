"""A width-matched control has to match the arm it controls for.

`build_random_control` hardcodes one arm's shape: two sources, one antibody-only
and one covering both sides, both 512 wide. That is the AntiBERTy + ESM-IF1 arm.
Using it for an arm with a single 128-wide source covering both sides would
compare arms of different widths and different parameter counts, and the
difference between them would no longer be only what the columns contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from interaction_first.composite_embeddings import (  # noqa: E402
    RepresentationError,
    RepresentationSource,
    build_matched_random_control,
)


def source(name: str, dim: int, keys):
    return RepresentationSource(
        name=name, root=Path("/nonexistent"), dim=dim,
        parts_by_key={key: (key,) for key in keys},
    )


def test_the_control_takes_each_source_width():
    real = [source("proteinmpnn", 128, ["ab1", "ag1", "ab2", "ag2"])]
    control = build_matched_random_control(Path("m.csv"), real)
    # `width` adds the presence columns; `dim` is the projection itself.
    assert [c.dim for c in control] == [128]
    assert [c.width for c in control] == [real[0].width]


def test_the_control_takes_each_source_coverage():
    real = [source("proteinmpnn", 128, ["ab1", "ag1"])]
    control = build_matched_random_control(Path("m.csv"), real)
    assert control[0].covers("ab1") and control[0].covers("ag1")
    assert not control[0].covers("something_else")


def test_an_antibody_only_source_yields_an_antibody_only_control():
    """The asymmetry has to survive: AntiBERTy says nothing about an antigen."""
    real = [source("antiberty", 512, ["ab1", "ab2"]),
            source("esm_if1", 512, ["ab1", "ab2", "ag1", "ag2"])]
    control = build_matched_random_control(Path("m.csv"), real)
    assert not control[0].covers("ag1"), "the antibody-only source stays antibody-only"
    assert control[1].covers("ag1")


def test_one_control_source_per_real_source():
    real = [source("a", 64, ["k"]), source("b", 128, ["k"]), source("c", 256, ["k"])]
    control = build_matched_random_control(Path("m.csv"), real)
    assert [c.dim for c in control] == [64, 128, 256]


def test_each_control_source_gets_its_own_seed():
    """Otherwise two 128-wide sources would carry identical random columns."""
    torch = pytest.importorskip("torch")
    real = [source("a", 128, ["k"]), source("b", 128, ["k"])]
    control = build_matched_random_control(Path("m.csv"), real, seed=7)
    base = torch.randn(5, 64)
    first = control[0].features("k", 5, "cpu", base)
    second = control[1].features("k", 5, "cpu", base)
    assert not bool(torch.equal(first, second)), "distinct sources need distinct draws"


def test_an_empty_arm_has_no_control():
    with pytest.raises(RepresentationError, match="at least one real source"):
        build_matched_random_control(Path("m.csv"), [])
