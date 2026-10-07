import argparse
import logging
import os
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_symlink(source_path: Path, target_path: Path):
    """Creates a symlink to source_path at target_path."""
    if not source_path.exists():
        logger.debug(f"Source file not found: {source_path}")
        return False
    
    if target_path.exists():
        return True
        
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a temporary file for atomic symlink creation
    tmp_path = target_path.with_name(target_path.name + ".tmp")
    try:
        os.symlink(source_path, tmp_path)
        tmp_path.rename(target_path)
        return True
    except FileExistsError:
        if tmp_path.exists():
            tmp_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed to symlink {source_path} to {target_path}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False

def main():
    parser = argparse.ArgumentParser(description="Organize SAbDab2 embeddings into two-character prefix subdirectories by symlinking.")
    parser.add_argument("--instances", type=Path, required=True, help="Path to instances.parquet")
    parser.add_argument("--source-ag", type=Path, required=True, help="Path to source ESM-2 antigen embeddings directory")
    parser.add_argument("--source-ab", type=Path, required=True, help="Path to source AntiBERTy antibody embeddings directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to new organized embeddings dir")
    
    args = parser.parse_args()
    
    if not args.instances.exists():
        logger.error(f"Instances file not found: {args.instances}")
        return
        
    logger.info(f"Loading instances from {args.instances}")
    try:
        df = pd.read_parquet(args.instances)
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        return
    
    if "ab_sequence_sha256" not in df.columns or "ag_sequence_sha256" not in df.columns:
        logger.error("instances.parquet must contain 'ab_sequence_sha256' and 'ag_sequence_sha256' columns.")
        return
        
    ab_hashes = df["ab_sequence_sha256"].dropna().unique()
    ag_hashes = df["ag_sequence_sha256"].dropna().unique()
    
    logger.info(f"Found {len(ab_hashes)} unique antibody hashes and {len(ag_hashes)} unique antigen hashes.")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    missing_ab = 0
    missing_ag = 0
    
    logger.info("Processing antibody embeddings...")
    for h in ab_hashes:
        source_path = args.source_ab / h[:2] / f"{h}.pt"
        target_path = args.output_dir / h[:2] / f"{h}.pt"
        if not create_symlink(source_path, target_path):
            missing_ab += 1
            
    logger.info("Processing antigen embeddings...")
    for h in ag_hashes:
        source_path = args.source_ag / h[:2] / f"{h}.pt"
        target_path = args.output_dir / h[:2] / f"{h}.pt"
        if not create_symlink(source_path, target_path):
            missing_ag += 1
            
    if missing_ab > 0:
        logger.warning(f"Missing {missing_ab}/{len(ab_hashes)} antibody embeddings.")
    if missing_ag > 0:
        logger.warning(f"Missing {missing_ag}/{len(ag_hashes)} antigen embeddings.")
        
    logger.info("Finished organizing embeddings.")

if __name__ == "__main__":
    main()
