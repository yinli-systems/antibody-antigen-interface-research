from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "remote_tools"
    / "abpairfm_multimodal_pretraining_v1"
    / "audit_pinder_holo_predicted_sequence_alignment_v1.py"
)
SPEC = importlib.util.spec_from_file_location("alignment_audit", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeAlignment:
    def __init__(self, target_blocks, query_blocks):
        self.aligned = (target_blocks, query_blocks)


class FakeAligner:
    def __init__(self, alignments):
        self.alignments = alignments

    def align(self, predicted: str, holo: str):
        return list(self.alignments)


def test_exact_mapping_masks_x_and_mismatch() -> None:
    alignment = FakeAlignment([[0, 4]], [[0, 4]])
    assert MODULE.exact_mapping(alignment, "ACDE", "ACXE") == (0, 1, -1, 3)
    assert MODULE.exact_mapping(alignment, "ACDE", "ACFE") == (0, 1, -1, 3)


def test_align_association_fails_closed_on_distinct_optimum() -> None:
    aligner = FakeAligner(
        [FakeAlignment([[0, 2]], [[0, 2]]), FakeAlignment([[1, 3]], [[0, 2]])]
    )
    mapping, ambiguous = MODULE.align_association(aligner, "AAA", "AA")
    assert ambiguous is True
    assert mapping == (-1, -1)


def _write_inventory(path: Path) -> None:
    fields = [
        "id",
        "predicted_R_pdb",
        "predicted_L_pdb",
        "sequence_R",
        "sequence_L",
        "sequence_sha256_R",
        "sequence_sha256_L",
    ]
    sequence = "ACDEFGHI"
    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=fields, delimiter="\t", lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "right--left",
                        "predicted_R_pdb": "af__one.pdb",
                        "predicted_L_pdb": "af__two.pdb",
                        "sequence_R": sequence,
                        "sequence_L": sequence,
                        "sequence_sha256_R": sequence_sha,
                        "sequence_sha256_L": sequence_sha,
                    }
                )


def _write_structures(path: Path) -> None:
    fields = ["member", "sequence", "sequence_sha256"]
    sequence = "ACDEFGHI"
    sequence_sha = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for member in ("af__one.pdb", "af__two.pdb"):
            writer.writerow(
                {"member": member, "sequence": sequence, "sequence_sha256": sequence_sha}
            )


def test_run_writes_audited_alignment_gate(tmp_path: Path, monkeypatch) -> None:
    inventory = tmp_path / "inventory.tsv.gz"
    structures = tmp_path / "structures.tsv"
    protocol = tmp_path / "protocol.json"
    _write_inventory(inventory)
    _write_structures(structures)
    protocol.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        MODULE,
        "configure_aligner",
        lambda: FakeAligner([FakeAlignment([[0, 8]], [[0, 8]])]),
    )
    args = SimpleNamespace(
        inventory=inventory,
        inventory_sha256=MODULE.sha256_file(inventory),
        structure_manifest=structures,
        structure_manifest_sha256=MODULE.sha256_file(structures),
        protocol_amendment=protocol,
        protocol_amendment_sha256=MODULE.sha256_file(protocol),
        minimum_exact_mapped_residues=8,
        minimum_exact_coverage=0.5,
        minimum_both_usable_systems=1,
        minimum_median_coverage=0.9,
        output_manifest=tmp_path / "alignment.tsv.gz",
        output_result=tmp_path / "result.json",
    )
    result = MODULE.run(args)
    assert result["status"] == "PASS_SEQUENCE_ONLY_ALIGNMENT_GATE_PENDING_PARENT"
    assert result["counts"]["both_endpoint_usable_systems"] == 1
    assert json.loads(args.output_result.read_text())["test_access"] == {
        "bound_coordinates_read": False,
        "pinder_test_rows_read": False,
        "pinder_validation_rows_read": False,
        "resolved_residue_flags_read": False,
    }
