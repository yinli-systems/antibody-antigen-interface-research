"""Verify that protocol-registered source hashes still bind real repository bytes.

Frozen protocols under ``protocols/`` register the SHA-256 of the source files
they govern. Evidence is append-only, so an older protocol legitimately keeps an
older hash: a superseded registration is not a defect. What *is* a defect is a
protocol-governed source file for which **no** registration binds the bytes
currently in the tree. When that happens the governing protocol no longer
describes the code it claims to freeze, and every downstream receipt inherits
the gap silently.

This module extracts those bindings, resolves them against the correct source
root, and reports files whose current bytes are unregistered. A checked-in
baseline records drift that predates the check, so continuous integration can
fail on *new* drift without rewriting history.

Two registration shapes appear in the tracked protocols:

``keypath``
    ``{"source_files": {"scripts/run.py": "<sha256>"}}`` -- the JSON key path
    below the anchor spells the repository-relative file name.

``sibling``
    ``{"source": {"files": {"runner": {"path": "scripts/run.py",
    "sha256": "<sha256>"}}}}`` -- the hash sits beside exactly one string that
    names a file.

Both are resolved against every candidate source root (the repository root plus
each independently installable subproject), because subproject manifests such as
``framecontact`` record paths relative to their own root.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

_SHA256_TEXT = re.compile(r"\A[0-9a-f]{64}\Z")
_BINDABLE_SUFFIXES = frozenset(
    {".py", ".sh", ".sbatch", ".json", ".md", ".jsonl", ".tsv", ".csv", ".txt", ".toml"}
)

BASELINE_RELATIVE_PATH = "protocols/source_binding_drift_baseline_v1.json"
BASELINE_FORMAT_VERSION = "source_binding_drift_baseline_v1"


class ProtocolBindingError(RuntimeError):
    """Raised when the binding inventory cannot be built or read."""


@dataclasses.dataclass(frozen=True)
class Binding:
    """One protocol registration of one source file's SHA-256."""

    protocol: str
    key_path: str
    source_path: str
    source_root: str
    expected_sha256: str

    @property
    def registered_path(self) -> str:
        """Repository-relative path of the bound file."""
        if self.source_root == ".":
            return self.source_path
        return f"{self.source_root}/{self.source_path}"


@dataclasses.dataclass(frozen=True)
class DriftRecord:
    """One protocol-governed file whose current bytes are unregistered."""

    registered_path: str
    actual_sha256: str
    registered_sha256: tuple[str, ...]
    protocols: tuple[str, ...]

    def as_json(self) -> dict[str, object]:
        """Canonical, order-stable payload for the baseline file."""
        return {
            "registered_path": self.registered_path,
            "actual_sha256": self.actual_sha256,
            "registered_sha256": list(self.registered_sha256),
            "protocols": list(self.protocols),
        }


def source_roots(repository_root: Path) -> tuple[str, ...]:
    """Return every candidate source root, repository root first.

    A directory holding its own ``pyproject.toml`` is an independently
    installable subproject whose manifests record paths relative to itself.
    """
    roots = ["."]
    for pyproject in sorted(repository_root.glob("*/pyproject.toml")):
        roots.append(pyproject.parent.name)
    return tuple(roots)


def _resolve(repository_root: Path, roots: Sequence[str], candidate: str) -> tuple[str, str] | None:
    """Resolve a recorded path against the candidate roots, if it names a file."""
    if not candidate or candidate.startswith("/") or "\\" in candidate:
        return None
    if Path(candidate).suffix not in _BINDABLE_SUFFIXES:
        return None
    if ".." in Path(candidate).parts:
        return None
    for root in roots:
        resolved = repository_root / root / candidate if root != "." else repository_root / candidate
        if resolved.is_file():
            return root, candidate
    return None


def _walk(
    node: object,
    key_path: list[str],
    repository_root: Path,
    roots: Sequence[str],
    protocol: str,
) -> Iterator[Binding]:
    """Yield every resolvable path/SHA-256 registration below ``node``."""
    if isinstance(node, Mapping):
        siblings = [
            resolved
            for value in node.values()
            if isinstance(value, str) and (resolved := _resolve(repository_root, roots, value))
        ]
        for key, value in node.items():
            if isinstance(value, (Mapping, list)):
                yield from _walk(value, key_path + [key], repository_root, roots, protocol)
                continue
            if not isinstance(value, str) or not _SHA256_TEXT.match(value):
                continue
            located = None
            for start in range(len(key_path) + 1):
                joined = "/".join(key_path[start:] + [key])
                located = _resolve(repository_root, roots, joined)
                if located is not None:
                    break
            if located is None and len(siblings) == 1:
                located = siblings[0]
            if located is None:
                continue
            root, source_path = located
            yield Binding(
                protocol=protocol,
                key_path="/".join(key_path + [key]),
                source_path=source_path,
                source_root=root,
                expected_sha256=value,
            )
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, key_path + [f"[{index}]"], repository_root, roots, protocol)


def collect_bindings(repository_root: Path) -> tuple[Binding, ...]:
    """Extract every resolvable source binding from the tracked protocols."""
    roots = source_roots(repository_root)
    bindings: list[Binding] = []
    baseline = (repository_root / BASELINE_RELATIVE_PATH).resolve()
    for protocol in sorted(repository_root.glob("protocols/**/*.json")):
        # The baseline records observed digests; reading it back as a registration
        # would make every recorded file register its own drift.
        if protocol.resolve() == baseline:
            continue
        try:
            document = json.loads(protocol.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolBindingError(f"{protocol} is not readable JSON: {error}") from error
        relative = protocol.relative_to(repository_root).as_posix()
        bindings.extend(_walk(document, [], repository_root, roots, relative))
    return tuple(bindings)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_drift(repository_root: Path, bindings: Sequence[Binding]) -> tuple[DriftRecord, ...]:
    """Return files whose current bytes no protocol registers.

    A file passes as soon as one registration matches the bytes on disk, which
    keeps superseded registrations from being reported as defects.
    """
    grouped: dict[str, list[Binding]] = {}
    for binding in bindings:
        grouped.setdefault(binding.registered_path, []).append(binding)

    drift: list[DriftRecord] = []
    for registered_path, entries in sorted(grouped.items()):
        actual = _digest(repository_root / registered_path)
        registered = {entry.expected_sha256 for entry in entries}
        if actual in registered:
            continue
        drift.append(
            DriftRecord(
                registered_path=registered_path,
                actual_sha256=actual,
                registered_sha256=tuple(sorted(registered)),
                protocols=tuple(sorted({entry.protocol for entry in entries})),
            )
        )
    return tuple(drift)


def load_baseline(repository_root: Path) -> dict[str, dict[str, object]]:
    """Read the accepted-drift baseline, keyed by registered path."""
    path = repository_root / BASELINE_RELATIVE_PATH
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format_version") != BASELINE_FORMAT_VERSION:
        raise ProtocolBindingError(
            f"{BASELINE_RELATIVE_PATH} declares an unsupported format_version"
        )
    entries = {}
    for entry in document.get("accepted_drift", ()):
        entries[entry["registered_path"]] = entry
    return entries


def unbaselined_drift(
    drift: Sequence[DriftRecord],
    baseline: Mapping[str, Mapping[str, object]],
) -> tuple[DriftRecord, ...]:
    """Return drift that the baseline does not already record for these bytes.

    A baselined file that drifts *again* is reported, because the recorded
    ``actual_sha256`` no longer describes the tree.
    """
    fresh = []
    for record in drift:
        accepted = baseline.get(record.registered_path)
        if accepted is None or accepted.get("actual_sha256") != record.actual_sha256:
            fresh.append(record)
    return tuple(fresh)


def resolved_baseline_paths(
    drift: Sequence[DriftRecord],
    baseline: Mapping[str, Mapping[str, object]],
) -> tuple[str, ...]:
    """Return baseline entries whose file is no longer drifting."""
    drifting = {record.registered_path for record in drift}
    return tuple(sorted(set(baseline) - drifting))
