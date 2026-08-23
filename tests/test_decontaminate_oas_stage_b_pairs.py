from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from scripts import decontaminate_oas_stage_b_pairs as module


def _row(index: int) -> dict[str, str]:
    digest = f"{index:064x}"
    return {
        "heavy_sequence": "ACDEFGHIKLMNPQRSTVWY",
        "light_sequence": "YWVTSRQPNMLKIHGFEDCA",
        "cdr1_aa_heavy": "ACDE",
        "cdr2_aa_heavy": "FGHI",
        "cdr3_aa_heavy": "KLMN",
        "cdr1_aa_light": "YWVT",
        "cdr2_aa_light": "SRQP",
        "cdr3_aa_light": "NMLK",
        "study": "study",
        "run": "run",
        "species": "human",
        "donor_hash": f"{index + 10:064x}",
        "row_hash": digest,
    }


def _write_input(path: Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=module.REQUIRED_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(_row(1))
        writer.writerow(_row(2))


def test_export_and_mmseqs_hit_mapping_are_pair_safe(tmp_path: Path) -> None:
    source = tmp_path / "pairs.tsv.gz"
    fasta = tmp_path / "pairs.fasta"
    hits = tmp_path / "hits.tsv"
    _write_input(source)
    assert module.export_fasta(source, fasta, expected_rows=2) == 2
    text = fasta.read_text(encoding="ascii")
    assert text.count(">") == 4
    assert f">{1:064x}|heavy" in text
    hits.write_text(
        f"dev-heavy\t{1:064x}|heavy\t80\t20\t1\t1\t0\t100\n"
        f"dev-light\t{1:064x}|light\t75\t20\t1\t1\t0\t90\n",
        encoding="utf-8",
    )
    excluded, hit_count = module.excluded_pair_hashes(hits)
    assert excluded == {f"{1:064x}"}
    assert hit_count == 2


def test_mmseqs_target_chain_is_rejected_if_unknown(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        f"dev\t{1:064x}|antigen\t80\t20\t1\t1\t0\t100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target chain"):
        module.excluded_pair_hashes(hits)


def test_mmseqs_query_chain_mapping_supports_reversed_gpu_search(tmp_path: Path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text(
        f"{2:064x}|light\tdev-light\t75\t20\t1\t1\t0\t90\n",
        encoding="utf-8",
    )
    excluded, hit_count = module.excluded_pair_hashes(hits, oas_column="query")
    assert excluded == {f"{2:064x}"}
    assert hit_count == 1


def test_sbatch_keeps_test_sealed_and_uses_only_gpu_4090() -> None:
    path = (
        module.Path(__file__).resolve().parents[1]
        / "scripts"
        / "decontaminate_oas_stage_b_pairs.sbatch"
    )
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in text
    assert "hp_4090" not in text
    assert "#SBATCH --mem" not in text
    assert "/ssd/scxi253/model_deployment/envs/contact/bin/python3.10" in text
    assert "--min-seq-id 0.5 -c 0.8 --cov-mode 0" in text


def test_gpu_sbatch_reverses_search_and_uses_padded_target() -> None:
    path = (
        module.Path(__file__).resolve().parents[1]
        / "scripts"
        / "decontaminate_oas_stage_b_pairs_gpu_v2.sbatch"
    )
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH -p gpu_4090" in text
    assert "hp_4090" not in text
    assert "#SBATCH --mem" not in text
    assert "makepaddedseqdb" in text
    assert 'search "${WORK}/oas_db" "${WORK}/development_db_padded"' in text
    assert "--gpu 1" in text
    assert "--oas-column query" in text
