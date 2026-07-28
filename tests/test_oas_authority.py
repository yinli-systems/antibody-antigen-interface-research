from __future__ import annotations

import copy

import pytest

from antibody_pretraining.oas_authority import (
    OASAuthorityError,
    build_oas_authority_manifest,
    canonical_json_bytes,
    validate_oas_authority_manifest,
    verify_manifest_self_hash,
)


def _html(*, second_script_url: str | None = None) -> bytes:
    script_url_a = (
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
        "StudyA_2020/csv/run_a_Heavy_Bulk.csv.gz"
    )
    script_url_b = second_script_url or (
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
        "StudyB_2021/csv/run_b_Light_Bulk.csv.gz"
    )
    return f"""
    <html><body>
      <p>Your search yielded <b>30</b> unique sequences from <b>2</b> studies.</p>
      <script>
        var CSV = ["wget {script_url_b}", "wget {script_url_a}"];
      </script>
      <table id="results_table">
        <thead><tr><th>Details</th><th>DS Name</th><th>#Unique Sequences</th>
          <th>Organism</th><th>Chain</th></tr></thead>
        <tbody>
          <tr><td><a href="../dataunit?unit=StudyA_2020%2Fcsv%2Frun_a_Heavy_Bulk.csv.gz">Details</a></td>
            <td>StudyA_2020</td><td>10</td><td>human</td><td>Heavy</td></tr>
          <tr><td><a href="../dataunit?unit=StudyB_2021%2Fcsv%2Frun_b_Light_Bulk.csv.gz">Details</a></td>
            <td>StudyB_2021</td><td>20</td><td>human</td><td>Light</td></tr>
        </tbody>
      </table>
    </body></html>
    """.encode("utf-8")


def test_build_authority_cross_checks_script_table_and_counts() -> None:
    manifest = build_oas_authority_manifest(_html(), source_class="unpaired")

    assert manifest["summary"] == {
        "data_unit_count": 2,
        "headline_study_count": 2,
        "headline_unique_sequence_count": 30,
        "observed_study_count": 2,
        "table_unique_sequence_count_sum": 30,
    }
    assert manifest["studies"] == ["StudyA_2020", "StudyB_2021"]
    assert [entry["study"] for entry in manifest["data_units"]] == [
        "StudyA_2020",
        "StudyB_2021",
    ]
    assert verify_manifest_self_hash(manifest)
    validate_oas_authority_manifest(manifest)
    assert canonical_json_bytes(manifest).endswith(b"\n")


def test_authority_rejects_disagreement_between_independent_url_views() -> None:
    mismatched_url = (
        "https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/"
        "StudyB_2021/csv/different.csv.gz"
    )
    with pytest.raises(OASAuthorityError, match="Bulk script and table disagree"):
        build_oas_authority_manifest(
            _html(second_script_url=mismatched_url),
            source_class="unpaired",
        )


def test_authority_rejects_wrong_host_duplicate_and_count_tamper() -> None:
    wrong_host = _html().replace(
        b"https://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/StudyB_2021",
        b"https://attacker.example/webapps/ngsdb/unpaired/StudyB_2021",
        1,
    )
    with pytest.raises(OASAuthorityError, match="Invalid official OAS"):
        build_oas_authority_manifest(wrong_host, source_class="unpaired")

    duplicate = _html().replace(b"StudyB_2021/csv/run_b_Light", b"StudyA_2020/csv/run_a_Heavy")
    with pytest.raises(OASAuthorityError):
        build_oas_authority_manifest(duplicate, source_class="unpaired")

    wrong_count = _html().replace(b"<b>30</b>", b"<b>31</b>", 1)
    with pytest.raises(OASAuthorityError, match="count disagrees"):
        build_oas_authority_manifest(wrong_count, source_class="unpaired")


def test_manifest_self_hash_rejects_any_mutation() -> None:
    manifest = build_oas_authority_manifest(_html(), source_class="unpaired")
    mutated = copy.deepcopy(manifest)
    mutated["data_units"][0]["unique_sequences"] += 1

    assert not verify_manifest_self_hash(mutated)
    with pytest.raises(OASAuthorityError, match="self-hash"):
        validate_oas_authority_manifest(mutated)


def test_manifest_validator_rejects_rehashed_semantic_tamper() -> None:
    manifest = build_oas_authority_manifest(_html(), source_class="unpaired")
    manifest["data_units"][0]["canonical_url"] = manifest["data_units"][1][
        "canonical_url"
    ]
    unhashed = dict(manifest)
    del unhashed["manifest_sha256"]
    import hashlib

    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()

    with pytest.raises(OASAuthorityError, match="URL and path disagree"):
        validate_oas_authority_manifest(manifest)


def test_authority_rejects_unknown_source_class_and_malformed_html() -> None:
    with pytest.raises(OASAuthorityError, match="source_class"):
        build_oas_authority_manifest(_html(), source_class="unknown")
    with pytest.raises(OASAuthorityError, match="result-count headline"):
        build_oas_authority_manifest(b"<html></html>", source_class="unpaired")
