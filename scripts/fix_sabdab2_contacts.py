import os
import glob
import numpy as np

directory = "/ssd/scxi253/model_deployment/data/interaction_first_asep/contacts/"
files = glob.glob(directory + "*.npz")

for path in files:
    mat = np.load(path)
    ab_len = int(np.atleast_1d(mat["antibody_length"])[0])
    ag_len = int(np.atleast_1d(mat["antigen_length"])[0])
    
    # We only have positive edges right now
    ab_pos = mat["pair_antibody_local"]
    ag_pos = mat["pair_antigen_local"]
    
    # Create dense matrix of labels
    dense = np.zeros((ab_len, ag_len), dtype=np.uint8)
    if not (len(ab_pos) == 1 and ab_pos[0] == 0 and len(mat["pair_labels"]) == 0):
        dense[ab_pos, ag_pos] = 1
        
    # Cartesian product
    ab_idx, ag_idx = np.where(np.ones((ab_len, ag_len), dtype=bool))
    
    labels = dense[ab_idx, ag_idx]
    
    np.savez_compressed(
        path,
        format_version=np.array(["sabdab2_sparse_contact_v1"]),
        instance_id=mat["instance_id"],
        antibody_length=mat["antibody_length"],
        antigen_length=mat["antigen_length"],
        antibody_indices=np.arange(ab_len, dtype=np.int64),
        antigen_indices=np.arange(ag_len, dtype=np.int64),
        pair_antibody_local=ab_idx.astype(np.int64),
        pair_antigen_local=ag_idx.astype(np.int64),
        pair_labels=labels
    )
print(f"Fixed {len(files)} files!")
