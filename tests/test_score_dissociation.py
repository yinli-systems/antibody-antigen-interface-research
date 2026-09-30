from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extensions.frontier_multimodel_v1.score_dissociation import (
    dynamic_groups,
    expected_first_positive_reciprocal_rank,
    expected_recall_at_k,
    read_ca,
    safe_spearman,
)


def brute_first_reciprocal(scores, positives):
    indices = range(len(scores))
    values = []
    for order in itertools.permutations(indices):
        if any(scores[order[i]] < scores[order[i + 1]] for i in range(len(order) - 1)):
            continue
        first = min(i + 1 for i, index in enumerate(order) if positives[index])
        values.append(1.0 / first)
    return float(np.mean(values))


def brute_recall(scores, positives, k):
    indices = range(len(scores))
    values = []
    for order in itertools.permutations(indices):
        if any(scores[order[i]] < scores[order[i + 1]] for i in range(len(order) - 1)):
            continue
        values.append(float(any(positives[index] for index in order[:k])))
    return float(np.mean(values))


@pytest.mark.parametrize(
    "scores,positives",
    [
        ([3, 2, 1], [True, False, False]),
        ([1, 3, 2], [True, False, False]),
        ([1, 1, 1, 1], [True, False, False, False]),
        ([2, 2, 2, 1], [True, True, False, False]),
        ([3, 2, 2, 2], [False, True, False, True]),
    ],
)
def test_mrr_exact_tie_expectation_matches_enumeration(scores, positives):
    assert expected_first_positive_reciprocal_rank(scores, positives) == pytest.approx(
        brute_first_reciprocal(scores, positives)
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_recall_exact_tie_expectation_matches_enumeration(k):
    scores = [3, 2, 2, 2]
    positives = [False, True, False, True]
    assert expected_recall_at_k(scores, positives, k) == pytest.approx(
        brute_recall(scores, positives, k)
    )


def test_ranking_requires_a_positive():
    with pytest.raises(ValueError):
        expected_first_positive_reciprocal_rank([1, 2], [False, False])
    with pytest.raises(ValueError):
        expected_recall_at_k([1, 2], [False, False], 1)


def test_dynamic_groups_cover_every_key_once_and_obey_budget():
    lengths = {"a": 10, "b": 20, "c": 30, "d": 40, "e": 50}
    groups = list(dynamic_groups(list(lengths), lengths, fixed_length=7, max_batch=3, pair_budget=700))
    assert sorted(key for group in groups for key in group) == sorted(lengths)
    assert len({key for group in groups for key in group}) == len(lengths)
    for group in groups:
        assert len(group) <= 3
        assert max(lengths[key] for key in group) * 7 * len(group) <= 700


def test_dynamic_groups_allows_one_oversized_candidate_without_dropping_it():
    groups = list(dynamic_groups(["long"], {"long": 1000}, 1000, 8, 100))
    assert groups == [["long"]]


def test_read_ca_parses_only_ca_and_rejects_empty(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00100.00           N\n"
        "ATOM      2  CA  ALA A   1       4.000   5.000   6.000  1.00100.00           C\n"
        "ATOM      3  C   ALA A   1       7.000   8.000   9.000  1.00100.00           C\n"
    )
    assert np.array_equal(read_ca(pdb), np.asarray([[4, 5, 6]], dtype=np.float32))
    empty = tmp_path / "empty.pdb"
    empty.write_text("END\n")
    with pytest.raises(ValueError):
        read_ca(empty)


def test_safe_spearman_reports_monotone_and_constant_boundary():
    assert safe_spearman([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert safe_spearman([1, 2, 3], [8, 4, 2]) == pytest.approx(-1.0)
    assert safe_spearman([1, 1, 1], [1, 2, 3]) is None


def test_safe_spearman_drops_missing_proxy_pairs_without_imputation():
    assert safe_spearman([1, 2, 3, 4], [1, float("nan"), 3, 4]) == pytest.approx(1.0)
    assert safe_spearman([1, 2], [float("nan"), 2]) is None


def test_manifest_metadata_separates_sequences_from_artifact_keys(tmp_path: Path):
    import csv
    import hashlib
    from extensions.frontier_multimodel_v1.score_dissociation import manifest_metadata

    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["INSTANCE", "vh_seq", "vl_seq", "ag_seq"])
        writer.writeheader()
        writer.writerow({"INSTANCE": "x", "vh_seq": "ACD", "vl_seq": "EF", "ag_seq": "GHIK"})
    sequences, part_keys, pairs = manifest_metadata(path)
    ab_key = hashlib.sha256(b"ACDEF").hexdigest()
    ag_key = hashlib.sha256(b"GHIK").hexdigest()
    assert sequences[("antibody", ab_key)] == ("ACD", "EF")
    assert sequences[("antigen", ag_key)] == ("GHIK",)
    assert part_keys[("antibody", ab_key)] == (
        hashlib.sha256(b"ACD").hexdigest(), hashlib.sha256(b"EF").hexdigest()
    )
    assert pairs == {(ab_key, ag_key)}
    assert sum(map(len, sequences[("antibody", ab_key)])) == 5
