from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_string_v12_expanded_endpoint_union.py"


def module():
    spec = importlib.util.spec_from_file_location("string_endpoint_union", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_endpoint_inventory_is_unique_and_order_agnostic(tmp_path: Path) -> None:
    value = module()
    source = tmp_path / "edges.txt"
    source.write_text(
        " ".join(value.EXPECTED_HEADER) + "\n"
        "p2 p1 0 1 0 0 0 0 0 700\n"
        "p1 p3 0 1 0 0 0 0 0 701\n",
        encoding="utf-8",
    )
    compressed = tmp_path / "edges.zst"
    subprocess.run(["zstd", "-q", "-f", str(source), "-o", str(compressed)], check=True)
    assert value.endpoints(compressed) == ({"p1", "p2", "p3"}, 2)
