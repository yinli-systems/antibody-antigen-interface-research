import argparse
import logging
from pathlib import Path
import torch
import pandas as pd
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_embedding(source_path, target_path, expected_hash, expected_length):
    if target_path.exists():
        return True
    if not source_path.exists():
        return False
    
    try:
        data = torch.load(source_path, map_location="cpu", weights_only=True)
    except Exception:
        # Some are just tensors not state dicts, let's try standard load
        try:
            data = torch.load(source_path, map_location="cpu")
        except:
            return False
            
    # if it's already a dict with 'chunks', we are good
    if isinstance(data, dict) and "chunks" in data:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, target_path)
        return True
        
    # otherwise it's just a tensor of shape [L, D] or [1, L, D]
    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict) and "representations" in data: # ESM format
        tensor = data["representations"][list(data["representations"].keys())[0]]
    else:
        tensor = data
        
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0] # [L, D]
        
    # SAbDab2 ESM embeddings usually include BOS/EOS tokens.
    # But wait, AntiBERTy doesn't.
    # Let's just assume the tensor length matches expected_length. If it's length + 2, strip BOS/EOS.
    if tensor.shape[0] == expected_length + 2:
        tensor = tensor[1:-1]
        
    artifact = {
        "chunks": [tensor],
        "starts": [0],
        "ends": [expected_length],
        "sequence_length": expected_length,
        "sequence_sha256": expected_hash,
    }
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, target_path)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--source-ag", type=Path, required=True)
    parser.add_argument("--source-ab", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    
    df = pd.read_parquet(args.instances)
    
    missing_ab = 0
    missing_ag = 0
    
    for _, row in df.iterrows():
        ab_hash = row["ab_sequence_sha256"]
        ag_hash = row["ag_sequence_sha256"]
        ab_len = row["ab_length"]
        ag_len = row["ag_length"]
        
        # AB
        source_ab = args.source_ab / ab_hash[:2] / f"{ab_hash}.pt"
        target_ab = args.output_dir / ab_hash[:2] / f"{ab_hash}.pt"
        if not convert_embedding(source_ab, target_ab, ab_hash, ab_len):
            missing_ab += 1
            
        # AG
        source_ag = args.source_ag / ag_hash[:2] / f"{ag_hash}.pt"
        target_ag = args.output_dir / ag_hash[:2] / f"{ag_hash}.pt"
        if not convert_embedding(source_ag, target_ag, ag_hash, ag_len):
            missing_ag += 1
            
    logger.info(f"Missing AB: {missing_ab}, Missing AG: {missing_ag}")

if __name__ == "__main__":
    main()
