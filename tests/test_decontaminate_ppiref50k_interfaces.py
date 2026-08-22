from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/decontaminate_ppiref50k_interfaces_v2.py"


def module():
    spec = importlib.util.spec_from_file_location("ppiref_decontam", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_endpoint_ids_and_excluded_examples(tmp_path: Path) -> None:
    value = module()
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">1abc_A_B|1\nACDE\n>1abc_A_B|2\nFGHI\n", encoding="ascii")
    known = value.endpoint_ids(fasta)
    hits = tmp_path / "hits.tsv"
    hits.write_text("dev\t1abc_A_B|2\t0.6\t4\t0.9\t0.8\t1e-5\t20\n", encoding="utf-8")
    excluded, counts = value.excluded_examples(hits, known)
    assert excluded == {"1abc_A_B"}
    assert counts == {
        "development_queries_with_hits": 1,
        "excluded_endpoint_sequences": 1,
        "excluded_interfaces": 1,
        "homology_hit_rows": 1,
    }


def test_subthreshold_hit_fails_closed(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("dev\t1abc_A_B|1\t0.49\t4\t0.9\t0.8\t1e-5\t20\n", encoding="utf-8")
    with pytest.raises(ValueError, match="homology gate"):
        module().excluded_examples(hits, {"1abc_A_B|1"})


def test_filter_fasta_removes_both_partners(tmp_path: Path) -> None:
    value = module()
    source = tmp_path / "source.fasta"
    output = tmp_path / "output.fasta"
    source.write_text(
        ">1abc_A_B|1\nACDE\n>1abc_A_B|2\nFGHI\n"
        ">2xyz_C_D|1\nKLMN\n>2xyz_C_D|2\nPQRS\n",
        encoding="ascii",
    )
    counts = value.filter_fasta(source, output, {"1abc_A_B"})
    assert output.read_text(encoding="ascii") == ">2xyz_C_D|1\nKLMN\n>2xyz_C_D|2\nPQRS\n"
    assert counts == {
        "input_endpoint_records": 4,
        "output_endpoint_records": 2,
        "unique_output_endpoint_sequences": 2,
    }
