#!/usr/bin/env python3
"""Generate AntiBERTy CDR embeddings with safe CUDA-to-CPU conversion."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from antiberty import AntiBERTyRunner
from Bio import SeqIO


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    data_root = args.data_root.resolve()
    fasta_path = data_root / "sequences" / "asep_ab_ag_seqres_1723.fasta"
    graph_dir = data_root / "asepv1_interim_graphs"
    output_path = data_root / "antibody" / "antiberty_embeddings" / "asep_antiberty_embeddings.pt"
    report_path = output_path.with_suffix(".report.json")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this job")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    runner = AntiBERTyRunner()
    print(f"device={runner.device} ({torch.cuda.get_device_name(device)})", flush=True)

    embeddings = {}
    if output_path.exists():
        embeddings.update(torch.load(output_path, map_location="cpu", weights_only=False))
        print(f"resuming_from={len(embeddings)}", flush=True)

    skipped = []
    errors = []
    completed_since_save = 0
    sequences = list(SeqIO.parse(str(fasta_path), "fasta"))
    for record in sequences:
        pdb_id = record.id.split("|")[0]
        if pdb_id == "5ies_0P" or pdb_id in embeddings:
            continue
        parts = str(record.seq).split(":")
        graph_path = graph_dir / f"{pdb_id}.pt"
        if len(parts) < 2 or not graph_path.exists():
            skipped.append({"pdb_id": pdb_id, "reason": "invalid_fasta_or_missing_graph"})
            continue
        try:
            antibody_sequence = parts[0] + parts[1]
            residue_embedding = runner.embed([antibody_sequence])[0][1:-1].detach().float().cpu()
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            cdr_mask = torch.as_tensor(graph["mapping"]["ab"]["seqres2cdr"], dtype=torch.bool)
            if residue_embedding.shape[0] != cdr_mask.shape[0]:
                raise ValueError(f"embedding/mask length mismatch: {residue_embedding.shape[0]} != {cdr_mask.shape[0]}")
            embeddings[pdb_id] = residue_embedding[cdr_mask].numpy()
            completed_since_save += 1
            if completed_since_save >= args.checkpoint_every:
                atomic_torch_save(embeddings, output_path)
                completed_since_save = 0
                print(f"saved={len(embeddings)}/{len(sequences)}", flush=True)
        except Exception as exc:
            errors.append({"pdb_id": pdb_id, "reason": repr(exc)})
            print(f"ERROR {pdb_id}: {exc!r}", flush=True)

    atomic_torch_save(embeddings, output_path)
    torch.cuda.synchronize()
    report = {
        "status": "complete" if not errors else "complete_with_errors",
        "model": "AntiBERTy_md_smooth",
        "device": torch.cuda.get_device_name(0),
        "input_records": len(sequences),
        "embedded_complexes": len(embeddings),
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "output": str(output_path),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
