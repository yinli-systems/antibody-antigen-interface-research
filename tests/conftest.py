"""Configure imports and portability for the curated repository layout.

Some tests cannot run from a fresh clone, for reasons that are not
interchangeable. Collapsing them into one red suite destroys the only signal
that matters: whether a change broke something. This module separates them by
recording, per test, the precondition it needs, so a test is skipped exactly
when its precondition is unmet and runs -- and must pass -- everywhere else.

``tests/portability_registry_v1.json`` holds one entry per affected test:

``requires``
    Preconditions. ``path`` names a repository-relative file that must exist,
    ``executable`` a program on ``PATH``, ``module_version`` an exact installed
    version, and ``absent_module`` a module that must *not* already be imported
    (some audits prove independence by refusing to run in a polluted process).
    An unmet precondition skips the test with the specific reason.

``expected_failure``
    The test binds a source digest that no protocol currently registers -- a
    live alarm for an entry in
    ``protocols/source_binding_drift_baseline_v1.json``. Marked non-strict
    ``xfail``, so it stays visible and is reported as ``XPASS`` the moment a
    numbered supersession reconciles it.

Nothing here disables a test. An entry that matches no collected test raises,
so the registry cannot rot as tests are renamed, and a precondition that is met
puts the test straight back into the suite.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_directory in (REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "src"):
    source_path = str(source_directory)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

REGISTRY_PATH = Path(__file__).resolve().parent / "portability_registry_v1.json"
REGISTRY_FORMAT_VERSION = "test_portability_registry_v1"

PORTABILITY_MARKER = "portability_registered"
DRIFT_MARKER = "known_source_drift"


def _load_registry() -> dict[str, object]:
    if not REGISTRY_PATH.exists():
        return {"entries": []}
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if document.get("format_version") != REGISTRY_FORMAT_VERSION:
        raise RuntimeError(f"{REGISTRY_PATH.name} declares an unsupported format_version")
    return document


def _unmet(requirement: dict[str, object]) -> str | None:
    """Return why one precondition is unmet, or ``None`` when it holds."""
    if "path" in requirement:
        relative = str(requirement["path"])
        if not (REPOSITORY_ROOT / relative).exists():
            return f"{relative} is not present"
        return None
    if "executable" in requirement:
        name = str(requirement["executable"])
        if shutil.which(name) is None:
            return f"the {name} executable is not on PATH"
        return None
    if "module_version" in requirement:
        name, expected = requirement["module_version"]
        try:
            observed = importlib.metadata.version(str(name))
        except importlib.metadata.PackageNotFoundError:
            return f"{name} is not installed"
        if observed != expected:
            return f"{name} {expected} is required but {observed} is installed"
        return None
    if "absent_module" in requirement:
        name = str(requirement["absent_module"])
        if any(loaded == name or loaded.startswith(f"{name}.") for loaded in sys.modules):
            return f"{name} is already imported in this process"
        return None
    raise RuntimeError(f"{REGISTRY_PATH.name}: unknown requirement {requirement!r}")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{PORTABILITY_MARKER}: has a recorded precondition in {REGISTRY_PATH.name}",
    )
    config.addinivalue_line(
        "markers",
        f"{DRIFT_MARKER}: binds a source digest no protocol currently registers",
    )


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    entries = _load_registry().get("entries", ())
    if not entries:
        return

    by_prefix = {entry["nodeid_prefix"]: entry for entry in entries}
    matched: set[str] = set()

    for item in items:
        entry = by_prefix.get(item.nodeid)
        if entry is None:
            # Class-level and parametrised ids extend the recorded prefix.
            for prefix, candidate in by_prefix.items():
                if item.nodeid.startswith(f"{prefix}::") or item.nodeid.startswith(f"{prefix}["):
                    entry = candidate
                    break
        if entry is None:
            continue

        matched.add(entry["nodeid_prefix"])
        item.add_marker(getattr(pytest.mark, PORTABILITY_MARKER))

        if entry.get("expected_failure"):
            reason = f"{entry['reason']}: {entry['detail']}"
            if entry["reason"] == DRIFT_MARKER:
                item.add_marker(getattr(pytest.mark, DRIFT_MARKER))
                reason += " (protocols/source_binding_drift_baseline_v1.json)"
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
            continue

        for requirement in entry.get("requires", ()):
            unmet = _unmet(requirement)
            if unmet is not None:
                item.add_marker(pytest.mark.skip(reason=f"{entry['reason']}: {unmet}"))
                break

    if _is_full_collection(config):
        stale = sorted(set(by_prefix) - matched)
        if stale:
            raise RuntimeError(
                f"{REGISTRY_PATH.name} lists {len(stale)} entry/entries matching no collected "
                f"test; remove them: {', '.join(stale[:5])}"
            )


def _is_full_collection(config: pytest.Config) -> bool:
    """Only a whole-suite run can prove a registry entry is stale."""
    if config.option.keyword or config.option.markexpr:
        return False
    return not config.option.file_or_dir
