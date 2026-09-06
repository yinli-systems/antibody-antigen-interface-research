import argparse
import hashlib
import logging
import os
import tempfile
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def hash_sequence(seq: str) -> str:
    """Computes the SHA-256 hash of a sequence string."""
    if not isinstance(seq, str) or not seq:
        return ""
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()

def prepare_instances(manifest_path: Path, output_path: Path) -> None:
    logger.info(f"Loading manifest from {manifest_path}")
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest file: {e}")
        raise

    required_columns = ['INSTANCE', 'ag_seq', 'ab_ag_cluster', 'split']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")

    logger.info("Processing data...")
    # Rename columns to match the requested output
    df = df.rename(columns={
        'INSTANCE': 'instance_id',
        'ab_ag_cluster': 'cluster'
    })
    
    # Create ab_seq
    df['vh_seq'] = df['vh_seq'].fillna('')
    df['vl_seq'] = df['vl_seq'].fillna('')
    df['ab_seq'] = df['vh_seq'] + df['vl_seq']

    # Compute hashes
    logger.info("Computing SHA-256 hashes for sequences...")
    df['ab_sequence_sha256'] = df['ab_seq'].apply(hash_sequence)
    df['ag_sequence_sha256'] = df['ag_seq'].apply(hash_sequence)
    df['ab_length'] = df['ab_seq'].str.len()
    df['ag_length'] = df['ag_seq'].str.len()

    # Select and reorder columns
    output_columns = [
        'instance_id', 'ab_seq', 'ag_seq', 
        'ab_sequence_sha256', 'ag_sequence_sha256', 
        'ab_length', 'ag_length',
        'split', 'cluster'
    ]
    
    out_df = df[output_columns].copy()
    # out_df.set_index('instance_id', inplace=True)

    logger.info(f"Writing output to {output_path}")
    
    # Atomic write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet")
    os.close(temp_fd)
    
    temp_path_obj = Path(temp_path)
    try:
        out_df.to_parquet(temp_path_obj)
        temp_path_obj.replace(output_path)
        logger.info(f"Successfully wrote {len(out_df)} instances to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write parquet file: {e}")
        if temp_path_obj.exists():
            temp_path_obj.unlink()
        raise

def main():
    parser = argparse.ArgumentParser(description="Prepare SAbDab2 instances parquet file from manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to input manifest CSV")
    parser.add_argument("--output", type=Path, required=True, help="Path to output instances.parquet")
    
    args = parser.parse_args()
    
    try:
        prepare_instances(args.manifest, args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
