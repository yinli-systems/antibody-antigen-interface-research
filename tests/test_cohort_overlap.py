"""Structural overlap between two cohorts, measured by PDB accession.

The question this answers is whether a model trained on one cohort has already
seen the structures another cohort will test it on. Identifier schemes differ
between cohorts and sequence digests cannot answer it either, because cohorts
select different residues -- one stores a full variable region, another only the
CDRs, so the same structure hashes differently in each. The accession is what
survives both.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location(
    "measure_cohort_overlap_under_test",
    REPOSITORY_ROOT / "scripts" / "measure_cohort_overlap.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

OverlapError = _MODULE.OverlapError
compare = _MODULE.compare
load_cohort = _MODULE.load_cohort
pdb_code = _MODULE.pdb_code


def test_both_identifier_schemes_resolve_to_the_same_accession() -> None:
    """The two cohorts in this project name the same structure differently."""
    assert pdb_code("pdb_00001j1p_H_L") == "1j1p"
    assert pdb_code("1j1p_0P") == "1j1p"


def test_the_zero_padding_is_stripped_not_taken_as_the_code() -> None:
    """Taking the first four characters of the padded field yields '0000'."""
    assert pdb_code("pdb_00001nbz_B_A") == "1nbz"
    assert pdb_code("pdb_00001nbz_B_A") != "0000"


def test_accessions_are_lowercased_so_the_schemes_meet() -> None:
    assert pdb_code("1A14_0P") == "1a14"


def test_an_unrecognised_identifier_returns_none_rather_than_a_guess() -> None:
    assert pdb_code("something-else") is None


def _cohort(train, validation, test) -> dict[str, set[str]]:
    return {"train": set(train), "validation": set(validation), "test": set(test)}


def test_train_to_test_overlap_is_flagged_as_contaminating() -> None:
    """The failure this exists for: a benchmark number from a seen structure."""
    left = _cohort(["1abc", "2abc"], [], ["9abc"])
    right = _cohort(["3abc"], ["4abc"], ["1abc", "5abc"])

    pairs = compare(left, right)

    entry = pairs["train_vs_test"]
    assert entry["shared_pdb_count"] == 1
    assert entry["share_of_right"] == pytest.approx(0.5)
    assert entry["contaminating"] is True


def test_train_to_train_overlap_is_not_flagged() -> None:
    """Sharing training structures wastes data; it does not invalidate a claim."""
    left = _cohort(["1abc"], [], ["9abc"])
    right = _cohort(["1abc"], ["4abc"], ["5abc"])

    assert compare(left, right)["train_vs_train"]["contaminating"] is False


def test_the_share_is_of_the_evaluated_cohort_not_the_training_one() -> None:
    """"23% of the test set was seen" is the sentence that matters."""
    left = _cohort([f"{n}abc" for n in range(100)], [], [])
    right = _cohort([], [], ["0abc", "1abc", "50zzz", "51zzz"])

    assert compare(left, right)["train_vs_test"]["share_of_right"] == pytest.approx(0.5)


def test_overlap_is_reported_in_both_directions(tmp_path: Path) -> None:
    left = _cohort(["1abc"], [], ["2abc"])
    right = _cohort(["2abc"], [], ["1abc"])

    assert compare(left, right)["train_vs_test"]["shared_pdb_count"] == 1
    assert compare(right, left)["train_vs_test"]["shared_pdb_count"] == 1


def _write(tmp_path: Path, ids, splits) -> tuple[Path, Path]:
    import pandas

    instances = tmp_path / "instances.parquet"
    pandas.DataFrame({"instance_id": ids}).to_parquet(instances)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    return instances, manifest


def test_split_members_without_a_prepared_instance_are_ignored(tmp_path: Path) -> None:
    """A split may name instances that preparation dropped."""
    pytest.importorskip("pandas")
    instances, manifest = _write(
        tmp_path,
        ["1abc_0P", "2abc_0P"],
        {"train": ["1abc_0P", "9zzz_0P"], "validation": ["2abc_0P"], "test": ["2abc_0P"]},
    )

    cohort = load_cohort(instances, manifest)

    assert cohort["train"] == {"1abc"}


def test_an_unresolvable_identifier_raises_rather_than_understating(tmp_path: Path) -> None:
    """Dropping it silently would report less overlap than there is."""
    pytest.importorskip("pandas")
    instances, manifest = _write(
        tmp_path, ["weird-id"], {"train": ["weird-id"], "validation": [], "test": []}
    )

    with pytest.raises(OverlapError, match="no recognisable PDB accession"):
        load_cohort(instances, manifest)


def test_a_manifest_without_splits_raises(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    import pandas

    instances = tmp_path / "i.parquet"
    pandas.DataFrame({"instance_id": ["1abc_0P"]}).to_parquet(instances)
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"components": []}), encoding="utf-8")

    with pytest.raises(OverlapError, match="no 'splits' mapping"):
        load_cohort(instances, manifest)
