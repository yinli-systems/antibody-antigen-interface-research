"""Configure imports for the curated repository layout."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_directory in (REPOSITORY_ROOT / "scripts", REPOSITORY_ROOT / "src"):
    source_path = str(source_directory)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
