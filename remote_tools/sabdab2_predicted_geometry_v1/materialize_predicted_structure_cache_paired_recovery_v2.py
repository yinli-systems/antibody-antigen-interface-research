#!/usr/bin/env python3
"""Run the frozen materializer with case-exact paired-AF3 resolution.

The recovery view exposes one directory named exactly ``{instance_id}_{type}``
for every development paired-antibody instance.  This wrapper changes only the
directory lookup; all parsing, sequence, coordinate, finite, and receipt checks
remain those of the frozen materializer.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


SOURCE = Path(__file__).with_name("materialize_predicted_structure_cache.py")
SPEC = importlib.util.spec_from_file_location(
    "sabdab2_predicted_geometry_frozen_materializer", SOURCE
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import frozen materializer: {SOURCE}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def find_case_exact_paired_prediction(
    root: Path,
    instance_id: str,
    antibody_type: str,
    expected_sequences: tuple[str, str],
    directory_names: tuple[str, ...],
    cache: dict[Any, Any],
) -> Any:
    del directory_names
    name = f"{instance_id}_{antibody_type}"
    directory = root / name
    source = MODULE.cached_prediction(cache, directory)
    if source.structure.sequences != expected_sequences:
        raise MODULE.CacheError(
            f"case-exact paired AF3 sequence differs for {instance_id}: "
            f"expected lengths {[len(x) for x in expected_sequences]}, "
            f"observed lengths {[len(x) for x in source.structure.sequences]}"
        )
    return source


MODULE.find_paired_prediction = find_case_exact_paired_prediction


if __name__ == "__main__":
    MODULE.main()
