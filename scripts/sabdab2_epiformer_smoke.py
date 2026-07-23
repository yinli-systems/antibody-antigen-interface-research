#!/usr/bin/env python3
"""Run EpiFormer end-to-end inference on one frozen SAbDab2 test instance."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, Select


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids


def write_chains(cif_path, output_path, chain_ids):
    structure = MMCIFParser(QUIET=True).get_structure("complex", str(cif_path))
    available = {chain.id for chain in structure.get_chains()}
    missing = set(chain_ids) - available
    if missing:
        raise ValueError(f"missing chains {sorted(missing)}; available={sorted(available)}")
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ChainSelect(chain_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epiformer-code", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instance-id")
    args = parser.parse_args()

    frozen = args.benchmark_root / "data/nfs/yangwuyu/baseline-eval/frozen/sabdab2"
    structure_root = args.benchmark_root / "data/nfs/yangwuyu/baseline-eval/datasets/pdb"
    split = json.loads((frozen / "splits.json").read_text())["sabdab2"]["test"]
    instance_id = args.instance_id or split[0]
    if instance_id not in split:
        raise ValueError(f"{instance_id} is not in the frozen test split")

    instances = pd.read_parquet(frozen / "instances.parquet")
    row = instances.loc[instances.instance_id == instance_id]
    if len(row) != 1:
        raise ValueError(f"expected one row for {instance_id}, got {len(row)}")
    row = row.iloc[0]

    cif_path = structure_root / Path(row.pdb_path).name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    antigen_pdb = args.output_dir / f"{instance_id}_antigen.pdb"
    antibody_pdb = args.output_dir / f"{instance_id}_antibody.pdb"
    ag_chains = [x for x in str(row.ag_chain_ids).split(",") if x]
    ab_chains = [x for x in str(row.ab_chain_ids).split(",") if x and x != "+"]
    write_chains(cif_path, antigen_pdb, ag_chains)
    write_chains(cif_path, antibody_pdb, ab_chains)

    sys.path.insert(0, str(args.epiformer_code.resolve()))
    from inference import run_inference

    result = run_inference(
        antigen_pdb=str(antigen_pdb),
        antibody_pdb=str(antibody_pdb),
        checkpoint_path=str(args.checkpoint),
        threshold=0.3,
        device="cuda:0",
        skip_filtering=False,
        sasa_threshold=5.0,
    )
    serializable = result.copy()
    serializable["epitope_probabilities"] = result["epitope_probabilities"].tolist()
    serializable["epitope_predictions"] = result["epitope_predictions"].tolist()
    serializable["instance_id"] = instance_id
    serializable["canonical_antigen_length"] = len(row.ag_seq)
    serializable["canonical_antibody_length"] = len(row.ab_seq)
    (args.output_dir / "result.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "instance_id": instance_id,
        "antigen_chains": ag_chains,
        "antibody_chains": ab_chains,
        "predicted_surface_residues": len(result["epitope_probabilities"]),
        "predicted_epitopes": int(result["epitope_predictions"].sum()),
        "status": "complete",
    }, indent=2))


if __name__ == "__main__":
    main()
