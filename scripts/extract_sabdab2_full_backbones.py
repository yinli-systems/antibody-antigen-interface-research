import argparse
import logging
import os
import tempfile
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from Bio.PDB import PDBParser, MMCIFParser

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_backbone_coords(structure, chains, is_antigen=False):
    if isinstance(chains, str):
        if is_antigen:
            chains = [c.strip() for c in chains.replace('|', ',').split(',') if c.strip()]
        else:
            chains = [chains]
    if not chains:
        return None
    
    atoms_to_extract = ['N', 'CA', 'C']
    coords = []
    
    model = structure[0]
    for chain_id in chains:
        if chain_id not in model:
            logging.warning(f"Chain {chain_id} not found in structure.")
            continue
        chain = model[chain_id]
        for residue in chain:
            if residue.id[0] != ' ':  # skip heteroatoms
                continue
            
            has_all_atoms = all(atom in residue for atom in atoms_to_extract)
            if has_all_atoms:
                res_coords = [residue[atom].coord for atom in atoms_to_extract]
                coords.append(res_coords)

    if not coords:
        return torch.empty((0, 3, 3), dtype=torch.float32)
        
    coords = np.array(coords) # [L, 3, 3]
    return torch.tensor(coords, dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser(description="Extract N, CA, C coordinates from PDB/mmCIF files.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to canonical_manifest.csv")
    parser.add_argument("--pdb-dir", type=str, required=True, help="Path to directory containing PDB/mmCIF files")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pt file")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    pdb_dir = Path(args.pdb_dir)
    output_path = Path(args.output)
    
    if not manifest_path.exists():
        logging.error(f"Manifest file {manifest_path} not found.")
        return
        
    df = pd.read_csv(manifest_path)
    
    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser(QUIET=True)
    
    output_dict = {}
    
    for idx, row in df.iterrows():
        instance_id = row.get('instance_id', row.get('INSTANCE', str(idx)))
        pdb_id = row.get('PDB_ID', row.get('pdb_id', row.get('pdb', None)))
            
        if not pdb_id:
            logging.warning(f"Could not find pdb id for instance {instance_id}")
            continue
            
        pdb_id = pdb_id.lower()
        
        filepath = None
        
        # Check specific known subdirectories if necessary, or just rely on instance_id
        search_dirs = [
            pdb_dir, 
            pdb_dir / "pair-geometry-cif-cohort-v1"
        ]
        
        for search_dir in search_dirs:
            for ext in ['.pdb', '.cif', '.ent']:
                for prefix in [instance_id, pdb_id, f"pdb{pdb_id}"]:
                    cand = search_dir / f"{prefix}{ext}"
                    if cand.exists():
                        filepath = cand
                        break
                if filepath:
                    break
            if filepath:
                break
                
        if not filepath:
            logging.warning(f"Could not find structure file for {pdb_id} in {pdb_dir}")
            continue
            
        try:
            if filepath.suffix == '.cif':
                structure = cif_parser.get_structure(pdb_id, filepath)
            else:
                structure = pdb_parser.get_structure(pdb_id, filepath)
        except Exception as e:
            logging.error(f"Failed to parse {filepath}: {e}")
            continue
            
        h_chain = row.get('Hchain', 'H')
        if pd.isna(h_chain): h_chain = ''
        l_chain = row.get('Lchain', 'L')
        if pd.isna(l_chain): l_chain = ''
        ag_chains = row.get('agchains', row.get('antigen_chain', ''))
        if pd.isna(ag_chains): ag_chains = ''
        
        ab_chains = []
        if h_chain: ab_chains.append(h_chain)
        if l_chain: ab_chains.append(l_chain)
        
        ab_tensor = get_backbone_coords(structure, ab_chains, is_antigen=False)
        ag_tensor = get_backbone_coords(structure, ag_chains, is_antigen=True)
        
        if ab_tensor is None:
            ab_tensor = torch.empty((0, 3, 3), dtype=torch.float32)
        if ag_tensor is None:
            ag_tensor = torch.empty((0, 3, 3), dtype=torch.float32)
            
        output_dict[instance_id] = {
            'antibody_backbone': ab_tensor,
            'antigen_backbone': ag_tensor
        }
        
    logging.info(f"Processed {len(output_dict)} instances. Saving to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False, suffix='.pt') as tmp:
        tmp_name = tmp.name
        torch.save(output_dict, tmp_name)
    os.replace(tmp_name, output_path)
    logging.info("Done.")

if __name__ == "__main__":
    main()
