import json, os, sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio.PDB import MMCIFParser

ATOM_ORDER = ("N", "CA", "C")

def get_atom_coords(residue):
    coords = np.full((3, 3), np.nan, dtype=np.float32)
    if residue is None:
        return coords
    for idx, atom_name in enumerate(ATOM_ORDER):
        if residue.has_id(atom_name):
            val = residue[atom_name].get_coord()
            if np.isfinite(val).all():
                coords[idx] = val
    return coords

def main():
    split_path = Path("/vast/scxi253/antibody_workspace/sabdab2_455_split.json")
    manifest_path = Path("/ssd/scxi253/model_deployment/data/sabdab2-baseline-outsourcing/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/manifest.csv")
    mapping_dir = Path("/ssd/scxi253/model_deployment/data/sabdab2-baseline-outsourcing/data/nfs/yangwuyu/sabdab2_bipartite_contacts/canonical/mapping")
    cif_dir = Path("/ssd/scxi253/antibody_workspace/sabdab2_test_cifs")
    out_path = Path("/ssd/scxi253/antibody_workspace/results/sabdab2_455_backbones.pt")

    split = json.loads(split_path.read_text())
    test_ids = split["splits"]["test"]
    manifest = pd.read_csv(manifest_path).set_index("INSTANCE")

    parser = MMCIFParser(QUIET=True)
    cif_cache = {}

    backbones = {}
    success_count = 0
    fail_count = 0

    print(f"Starting extraction for {len(test_ids)} test instances...", flush=True)

    for i, inst in enumerate(test_ids):
        try:
            row = manifest.loc[inst]
            pdb_id = row["PDB_ID"]
            cif_path = cif_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                print(f"[{i+1}/{len(test_ids)}] Missing CIF: {cif_path}", flush=True)
                fail_count += 1
                continue

            if pdb_id not in cif_cache:
                structure = parser.get_structure(pdb_id, cif_path)
                model = next(structure.get_models())
                cif_cache[pdb_id] = model
            model = cif_cache[pdb_id]

            mapping_path = mapping_dir / f"{inst}.json"
            mapping = json.loads(mapping_path.read_text())

            # 1. Antibody Backbone
            ab_coords_list = []
            # Heavy Chain
            h_chain_name = str(row["Hchain"]).strip()
            if h_chain_name and h_chain_name not in ("+", "nan", "None") and h_chain_name in model:
                h_res = [r for r in model[h_chain_name] if r.has_id("CA")]
                vh_to_hseq = mapping.get("vh_to_hseq", [])
                for idx in vh_to_hseq:
                    if idx is not None and 0 <= idx < len(h_res):
                        ab_coords_list.append(get_atom_coords(h_res[idx]))
                    else:
                        ab_coords_list.append(np.full((3, 3), np.nan, dtype=np.float32))
            else:
                for _ in range(int(row["vh_len"])):
                    ab_coords_list.append(np.full((3, 3), np.nan, dtype=np.float32))

            # Light Chain
            l_chain_name = str(row["Lchain"]).strip()
            if l_chain_name and l_chain_name not in ("+", "nan", "None") and l_chain_name in model:
                l_res = [r for r in model[l_chain_name] if r.has_id("CA")]
                vl_to_lseq = mapping.get("vl_to_lseq", [])
                for idx in vl_to_lseq:
                    if idx is not None and 0 <= idx < len(l_res):
                        ab_coords_list.append(get_atom_coords(l_res[idx]))
                    else:
                        ab_coords_list.append(np.full((3, 3), np.nan, dtype=np.float32))
            else:
                for _ in range(int(row["vl_len"])):
                    ab_coords_list.append(np.full((3, 3), np.nan, dtype=np.float32))

            ab_coords = np.array(ab_coords_list, dtype=np.float32)
            expected_ab_len = int(row["vh_len"]) + int(row["vl_len"])
            if len(ab_coords) != expected_ab_len:
                print(f"Warning: {inst} ab len mismatch: {len(ab_coords)} vs {expected_ab_len}", flush=True)

            # 2. Antigen Backbone
            ag_coords_list = []
            ag_chain_name = str(mapping.get("agchains", row["agchains"])).strip()
            ag_res_dict = {}
            for ch in ag_chain_name.split():
                if ch in model:
                    for r in model[ch]:
                        if r.has_id("CA"):
                            ag_res_dict[len(ag_res_dict)] = r
            if not ag_res_dict and ag_chain_name in model:
                for r in model[ag_chain_name]:
                    if r.has_id("CA"):
                        ag_res_dict[len(ag_res_dict)] = r

            ag_to_agseq = mapping.get("ag_to_agseq", [])
            for idx in ag_to_agseq:
                if idx is not None and idx in ag_res_dict:
                    ag_coords_list.append(get_atom_coords(ag_res_dict[idx]))
                else:
                    ag_coords_list.append(np.full((3, 3), np.nan, dtype=np.float32))

            ag_coords = np.array(ag_coords_list, dtype=np.float32)
            expected_ag_len = int(row["ag_len"])
            if len(ag_coords) != expected_ag_len:
                print(f"Warning: {inst} ag len mismatch: {len(ag_coords)} vs {expected_ag_len}", flush=True)

            backbones[inst] = {
                "antibody_backbone": torch.from_numpy(ab_coords),
                "antigen_backbone": torch.from_numpy(ag_coords),
            }
            success_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(test_ids):
                print(f"Processed {i+1}/{len(test_ids)} instances (success: {success_count}, fail: {fail_count})", flush=True)
        except Exception as e:
            print(f"Error processing {inst}: {e}", flush=True)
            fail_count += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbones, out_path)
    print(f"\nExtraction complete! Saved {len(backbones)} backbones to {out_path}", flush=True)

if __name__ == "__main__":
    main()
