"""Tests for the protocol source-binding verifier.

The verifier itself is the guard that keeps frozen protocols bound to the source
they govern, so these tests pin its resolution rules, its append-only semantics,
and the baseline ratchet rather than only asserting that the tree is currently
clean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interaction_first.protocol_source_bindings import (
    BASELINE_FORMAT_VERSION,
    BASELINE_RELATIVE_PATH,
    Binding,
    collect_bindings,
    find_drift,
    load_baseline,
    resolved_baseline_paths,
    source_roots,
    unbaselined_drift,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / BASELINE_RELATIVE_PATH


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture_repository(tmp_path: Path) -> Path:
    """Build a miniature repository with one governed source file."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "runner.py").write_text("print('v1')\n", encoding="utf-8")
    return tmp_path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_keypath_and_sibling_registrations_are_both_resolved(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    digest = _digest(repository / "scripts" / "runner.py")
    _write(
        repository / "protocols" / "keypath.json",
        {"source_files": {"scripts/runner.py": digest}},
    )
    _write(
        repository / "protocols" / "sibling.json",
        {"source": {"files": {"runner": {"path": "scripts/runner.py", "sha256": digest}}}},
    )

    bindings = collect_bindings(repository)

    assert {binding.protocol for binding in bindings} == {
        "protocols/keypath.json",
        "protocols/sibling.json",
    }
    assert {binding.registered_path for binding in bindings} == {"scripts/runner.py"}
    assert find_drift(repository, bindings) == ()


def test_a_superseded_registration_is_not_drift(tmp_path: Path) -> None:
    """Append-only history keeps stale digests; one current digest clears the file."""
    repository = _fixture_repository(tmp_path)
    digest = _digest(repository / "scripts" / "runner.py")
    _write(repository / "protocols" / "v1.json", {"source_files": {"scripts/runner.py": "0" * 64}})
    _write(repository / "protocols" / "v2.json", {"source_files": {"scripts/runner.py": digest}})

    assert find_drift(repository, collect_bindings(repository)) == ()


def test_a_file_no_protocol_registers_is_reported(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    _write(repository / "protocols" / "v1.json", {"source_files": {"scripts/runner.py": "0" * 64}})

    drift = find_drift(repository, collect_bindings(repository))

    assert len(drift) == 1
    assert drift[0].registered_path == "scripts/runner.py"
    assert drift[0].registered_sha256 == ("0" * 64,)
    assert drift[0].protocols == ("protocols/v1.json",)


def test_subproject_manifests_resolve_against_their_own_root(tmp_path: Path) -> None:
    """A manifest may record paths relative to an installable subproject root."""
    repository = _fixture_repository(tmp_path)
    (repository / "framecontact" / "scripts").mkdir(parents=True)
    (repository / "framecontact" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    subproject_file = repository / "framecontact" / "scripts" / "train.py"
    subproject_file.write_text("print('subproject')\n", encoding="utf-8")
    _write(
        repository / "protocols" / "tree.json",
        {"records": [{"path": "scripts/train.py", "sha256": _digest(subproject_file)}]},
    )

    bindings = collect_bindings(repository)

    assert "framecontact" in source_roots(repository)
    assert [binding.registered_path for binding in bindings] == ["framecontact/scripts/train.py"]
    assert find_drift(repository, bindings) == ()


def test_the_baseline_is_not_read_back_as_a_registration(tmp_path: Path) -> None:
    """The baseline records observed digests; treating them as registrations
    would let every recorded file silently register its own drift."""
    repository = _fixture_repository(tmp_path)
    runner = repository / "scripts" / "runner.py"
    _write(repository / "protocols" / "v1.json", {"source_files": {"scripts/runner.py": "0" * 64}})
    _write(
        repository / BASELINE_RELATIVE_PATH,
        {
            "format_version": BASELINE_FORMAT_VERSION,
            "accepted_drift": [
                {
                    "registered_path": "scripts/runner.py",
                    "actual_sha256": _digest(runner),
                    "registered_sha256": ["0" * 64],
                    "protocols": ["protocols/v1.json"],
                }
            ],
        },
    )

    drift = find_drift(repository, collect_bindings(repository))

    assert len(drift) == 1
    assert unbaselined_drift(drift, load_baseline(repository)) == ()


def test_a_baselined_file_that_drifts_again_is_reported(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    runner = repository / "scripts" / "runner.py"
    _write(repository / "protocols" / "v1.json", {"source_files": {"scripts/runner.py": "0" * 64}})
    _write(
        repository / BASELINE_RELATIVE_PATH,
        {
            "format_version": BASELINE_FORMAT_VERSION,
            "accepted_drift": [
                {
                    "registered_path": "scripts/runner.py",
                    "actual_sha256": _digest(runner),
                    "registered_sha256": ["0" * 64],
                    "protocols": ["protocols/v1.json"],
                }
            ],
        },
    )
    runner.write_text("print('v2')\n", encoding="utf-8")

    drift = find_drift(repository, collect_bindings(repository))
    fresh = unbaselined_drift(drift, load_baseline(repository))

    assert [record.registered_path for record in fresh] == ["scripts/runner.py"]


def test_a_repaired_file_is_reported_as_a_stale_baseline_entry(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    digest = _digest(repository / "scripts" / "runner.py")
    _write(repository / "protocols" / "v1.json", {"source_files": {"scripts/runner.py": digest}})
    _write(
        repository / BASELINE_RELATIVE_PATH,
        {
            "format_version": BASELINE_FORMAT_VERSION,
            "accepted_drift": [
                {
                    "registered_path": "scripts/runner.py",
                    "actual_sha256": "0" * 64,
                    "registered_sha256": ["1" * 64],
                    "protocols": ["protocols/v1.json"],
                }
            ],
        },
    )

    drift = find_drift(repository, collect_bindings(repository))

    assert drift == ()
    assert resolved_baseline_paths(drift, load_baseline(repository)) == ("scripts/runner.py",)


def test_paths_outside_the_repository_are_never_bound(tmp_path: Path) -> None:
    repository = _fixture_repository(tmp_path)
    _write(
        repository / "protocols" / "escape.json",
        {
            "source_files": {
                "../outside.py": "0" * 64,
                "/etc/passwd": "1" * 64,
            }
        },
    )

    assert collect_bindings(repository) == ()


def test_binding_registered_path_joins_the_subproject_root() -> None:
    binding = Binding(
        protocol="protocols/x.json",
        key_path="source_files/a.py",
        source_path="a.py",
        source_root="framecontact",
        expected_sha256="0" * 64,
    )

    assert binding.registered_path == "framecontact/a.py"


def test_repository_baseline_is_well_formed_and_fully_triaged() -> None:
    """Every carried drift must name why it is carried and when it started."""
    document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert document["format_version"] == BASELINE_FORMAT_VERSION
    assert document["accepted_drift"], "an empty baseline should be deleted, not committed"
    for entry in document["accepted_drift"]:
        assert entry["classification"] in {
            "historical_source_tree_snapshot",
            "unreconciled_governance_drift",
        }
        assert entry["first_unregistered_in"]
        assert entry["registered_sha256"]
        assert entry["protocols"]


def test_repository_has_no_unbaselined_source_drift() -> None:
    """The guard this whole module exists to provide."""
    drift = find_drift(ROOT, collect_bindings(ROOT))
    fresh = unbaselined_drift(drift, load_baseline(ROOT))

    assert [record.registered_path for record in fresh] == []
