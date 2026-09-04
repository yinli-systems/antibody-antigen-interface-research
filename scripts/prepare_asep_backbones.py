#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import numpy as np
from Bio.PDB import PDBParser
from tqdm import tqdm
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from interaction_first.sabdab2_structure import _chain_backbone, _chain_lookup

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-extract sliced backbones from ASEP PDBs")
    parser.add_argument("--pdb-dir", type=Path, default=Path("/ssd/scxi253/model_deployment/data/asep/structures"))
    parser.add_argument("--graph-dir", type=Path, default=Path("/ssd/scxi253/model_deployment/data/asep/asepv1_interim_graphs"))
    parser.add_argument("--output", type=Path, default=Path("/ssd/scxi253/model_deployment/data/interaction_first_asep/backbones.pt"))
    return parser.parse_args()

def extract_labels_from_remark(pdb_path: Path):
    ab_labels = []
    ag_labels = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("REMARK 950 CHAIN"):
                parts = line.split()
                if len(parts) >= 6:
                    chain_type = parts[3]
                    label = parts[4]
                    if chain_type in ("H", "L"):
                        ab_labels.append(label)
                    elif chain_type == "A":
                        ag_labels.append(label)
    return ab_labels, ag_labels

def align_backbone(obs_seq: str, obs_coords: np.ndarray, full_seq: str):
    full_coords = np.full((len(full_seq), 3, 3), np.nan, dtype=np.float32)
    obs_idx = 0
    full_idx = 0
    while obs_idx < len(obs_seq) and full_idx < len(full_seq):
        if obs_seq[obs_idx] == full_seq[full_idx]:
            full_coords[full_idx] = obs_coords[obs_idx]
            obs_idx += 1
            full_idx += 1
        else:
            full_idx += 1
    if obs_idx < len(obs_seq):
        raise ValueError(f"Alignment failed: stuck at {obs_idx}/{len(obs_seq)}")
    return full_coords

def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdb_files = list(args.pdb_dir.glob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files.")

    parser = PDBParser(QUIET=True)
    backbones_dict = {}
    skipped = 0

    for pdb_path in tqdm(pdb_files, desc="Extracting backbones"):
        instance_id = pdb_path.stem
        graph_path = args.graph_dir / f"{instance_id}.pt"
        if not graph_path.exists():
            skipped += 1
            continue
        
        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            mapping = graph["mapping"]
            seqres = graph["seqres"]
        except Exception as e:
            skipped += 1
            continue

        ab_labels, ag_labels = extract_labels_from_remark(pdb_path)
        try:
            structure = parser.get_structure(instance_id, str(pdb_path))
            model = structure[0]
        except Exception as e:
            skipped += 1
            continue

        def get_observed(labels):
            backbones = []
            seqs = []
            for label in labels:
                try:
                    chain = _chain_lookup(model, label)
                    seq, coords = _chain_backbone(chain, None)
                    backbones.append(coords)
                    seqs.append(seq)
                except Exception:
                    pass
            if not backbones:
                return "", None
            return "".join(seqs), np.concatenate(backbones, axis=0)

        ab_obs_seq, ab_obs_bb = get_observed(ab_labels)
        ag_obs_seq, ag_obs_bb = get_observed(ag_labels)

        if not ab_obs_seq or not ag_obs_seq:
            skipped += 1
            continue
            
        ab_full_seq = "".join(seqres["ab"].values())
        ag_full_seq = "".join(seqres["ag"].values())

        try:
            ab_full_bb = align_backbone(ab_obs_seq, ab_obs_bb, ab_full_seq)
            ag_full_bb = align_backbone(ag_obs_seq, ag_obs_bb, ag_full_seq)
            
            ab_mask = mapping["ab"]["seqres2cdr"] == 1
            ag_mask = mapping["ag"]["seqres2surf"] == 1
            
            ab_bb_sliced = ab_full_bb[ab_mask]
            ag_bb_sliced = ag_full_bb[ag_mask]

            backbones_dict[instance_id] = {
                "antibody_backbone": torch.from_numpy(ab_bb_sliced).float(),
                "antigen_backbone": torch.from_numpy(ag_bb_sliced).float()
            }
        except Exception as e:
            skipped += 1
            continue

    print(f"Successfully processed {len(backbones_dict)} backbones. Skipped {skipped}.")
    torch.save(backbones_dict, args.output)
    print("Done!")

if __name__ == "__main__":
    main()
