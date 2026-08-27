#!/usr/bin/env python3
"""Label-free recovery for set-valued SAbDab2 antigen-type metadata."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "prepare_temporal_double_cold_v1.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("temporal_label_free_base_v1", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen base label-free implementation")
    module = importlib.util.module_from_spec(spec)
    # ProcessPoolExecutor pickles functions by module/name.  Register the
    # dynamically loaded frozen module so worker processes can import it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recovery_parse_instance(module, record: dict[str, object], cif_path: Path) -> dict[str, object]:
    from Bio.PDB import MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure(str(record["instance"]), cif_path)
    model = next(structure.get_models())
    antibody_chains = [str(record["hchain"])]
    light = str(record["lchain"])
    if light not in {"", "NA", "None", "nan"}:
        antibody_chains.append(light)

    declared = list(module.split_field(str(record["antigen_chain"])))
    types = list(module.split_field(str(record["antigen_type"])))
    if not declared:
        raise ValueError(f"{record['instance']} has no declared antigen chain")
    if len(types) == 1 and types[0].upper() == "PROTEIN":
        antigen_chains = declared
        selection = "single-protein-type-applies-to-all-declared-chains"
        ag_parts = [module.observed_chain(model, chain) for chain in antigen_chains]
    elif len(types) == len(declared):
        antigen_chains = [
            chain
            for chain, kind in zip(declared, types, strict=True)
            if kind.upper() == "PROTEIN"
        ]
        selection = "chain-aligned-antigen-types"
        ag_parts = [module.observed_chain(model, chain) for chain in antigen_chains]
    else:
        # SAbDab2 can emit a set of antigen types rather than one type per chain.
        # In that case select declared chains solely by observed amino-acid
        # residues.  This is intra-chain, label-free metadata handling.
        antigen_chains = []
        ag_parts = []
        for chain in declared:
            try:
                part = module.observed_chain(model, chain)
            except ValueError as error:
                message = str(error)
                if "has no protein residues" in message or "lacks declared chain" in message:
                    continue
                raise
            antigen_chains.append(chain)
            ag_parts.append(part)
        selection = "set-valued-types-resolved-by-observed-amino-acid-residues"
    if not antigen_chains:
        raise ValueError(f"{record['instance']} has no observed protein antigen chain")

    ab_parts = [module.observed_chain(model, chain) for chain in antibody_chains]
    ab_sequence = "".join(part[0] for part in ab_parts)
    ag_sequence = "".join(part[0] for part in ag_parts)
    return {
        "instance_id": str(record["instance"]),
        "pdb_id": str(record["pdb"]),
        "antibody_chain_ids": antibody_chains,
        "antigen_chain_ids": antigen_chains,
        "antigen_chain_selection_authority": selection,
        "ab_sequence": ab_sequence,
        "ag_sequence": ag_sequence,
        "ab_sequence_sha256": module.sha256_bytes(ab_sequence.encode("ascii")),
        "ag_sequence_sha256": module.sha256_bytes(ag_sequence.encode("ascii")),
        "ab_length": len(ab_sequence),
        "ag_length": len(ag_sequence),
        "ab_backbone_complete_count": sum(part[1] for part in ab_parts),
        "ag_backbone_complete_count": sum(part[1] for part in ag_parts),
        "cif_path": str(cif_path),
        "cif_sha256": module.sha256_file(cif_path),
    }


def main() -> None:
    plan_path = Path(os.environ["TEMPORAL_LABEL_FREE_RECOVERY_PLAN"])
    plan_sha = os.environ["TEMPORAL_LABEL_FREE_RECOVERY_PLAN_SHA256"]
    if sha256_file(plan_path) != plan_sha:
        raise ValueError("temporal recovery plan SHA-256 differs")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    module = load_base()
    if (
        plan.get("format_version") != "framecontact_temporal_label_free_recovery_plan_v3"
        or plan.get("status")
        != "FROZEN_AFTER_THREE_LABEL_FREE_INTERFACE_FAILURES_BEFORE_RECOVERY"
        or plan.get("recovery", {}).get("base_script_sha256") != sha256_file(BASE_PATH)
        or plan.get("recovery", {}).get("wrapper_script_sha256")
        != sha256_file(Path(__file__))
        or plan.get("audit", {}).get("task_labels_read") is not False
        or plan.get("audit", {}).get("cross_chain_distances_computed") is not False
        or plan.get("cif_transfer_amendment", {}).get("status")
        != "PASS_CASE_SENSITIVE_TRANSFER_CORRECTION_FROZEN"
    ):
        raise ValueError("temporal recovery authority differs")
    module.parse_instance = lambda record, cif_path: recovery_parse_instance(
        module, record, cif_path
    )
    module.main()

    # The frozen base writes these create-only files.  Bind them to the recovery
    # authority without changing the original protocol field in the manifest.
    output_root = None
    for index, value in enumerate(os.sys.argv):
        if value == "--output-root" and index + 1 < len(os.sys.argv):
            output_root = Path(os.sys.argv[index + 1])
            break
    if output_root is None:
        raise ValueError("missing --output-root")
    manifest_path = output_root / "TEMPORAL_DOUBLE_COLD_LABEL_FREE_MANIFEST_V1.json"
    receipt_path = output_root / "TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECEIPT_V1.json"
    recovery_receipt_path = (
        output_root / "TEMPORAL_DOUBLE_COLD_LABEL_FREE_RECOVERY_RECEIPT_V1.json"
    )
    recovery_receipt = {
        "format_version": "framecontact_temporal_label_free_recovery_receipt_v1",
        "status": "PASS_LABEL_FREE_INTERFACE_RECOVERY_BOUND",
        "recovery_plan_sha256": plan_sha,
        "label_free_manifest_sha256": sha256_file(manifest_path),
        "label_free_receipt_sha256": sha256_file(receipt_path),
        "base_script_sha256": sha256_file(BASE_PATH),
        "wrapper_script_sha256": sha256_file(Path(__file__)),
        "semantic_change": plan["recovery"]["semantic_change"],
        "gate_thresholds_changed": False,
        "task_labels_read": False,
        "cross_chain_distances_computed": False,
        "selection_performed": False,
    }
    module.atomic_write(recovery_receipt_path, module.canonical_bytes(recovery_receipt))
    print(json.dumps(recovery_receipt, sort_keys=True))


if __name__ == "__main__":
    main()
