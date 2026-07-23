#!/usr/bin/env python3
"""Generate only the ESM-2 650M antigen embeddings required by EpiFormer."""

import argparse
import json
import os
import time
from pathlib import Path

import esm
import torch
from Bio import SeqIO


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def embed_sequence(model, batch_converter, sequence: str, device, max_residues=1022, overlap=128):
    """Embed a sequence exactly when possible, otherwise use averaged overlapping windows."""
    length = len(sequence)
    if length <= max_residues:
        starts = [0]
    else:
        step = max_residues - overlap
        starts = list(range(0, length - max_residues + 1, step))
        final_start = length - max_residues
        if starts[-1] != final_start:
            starts.append(final_start)

    accumulated = None
    counts = torch.zeros((length, 1), dtype=torch.float32)
    for start in starts:
        chunk = sequence[start:start + max_residues]
        _, _, tokens = batch_converter([("antigen", chunk)])
        tokens = tokens.to(device)
        with torch.inference_mode():
            result = model(tokens, repr_layers=[33], return_contacts=False)
            chunk_embedding = result["representations"][33][0, 1:-1].float().cpu()
        if accumulated is None:
            accumulated = torch.zeros((length, chunk_embedding.shape[1]), dtype=torch.float32)
        accumulated[start:start + len(chunk)] += chunk_embedding
        counts[start:start + len(chunk)] += 1
    return accumulated / counts, len(starts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    data_root = args.data_root.resolve()
    fasta_path = data_root / "sequences" / "asep_ab_ag_seqres_1723.fasta"
    graph_dir = data_root / "asepv1_interim_graphs"
    output_path = data_root / "antigen" / "plm_embeddings" / "ag_esm2_embeddings_asep.pt"
    report_path = output_path.with_suffix(".report.json")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this job")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()

    print(f"device={torch.cuda.get_device_name(device)}", flush=True)
    print(f"fasta={fasta_path}", flush=True)
    print("Loading ESM-2 650M only", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    payload = {"esm2_650m": {}, "esm2_3b": {}}
    if output_path.exists():
        previous = torch.load(output_path, map_location="cpu", weights_only=False)
        payload["esm2_650m"].update(previous.get("esm2_650m", {}))
        print(f"resuming_from={len(payload['esm2_650m'])}", flush=True)

    skipped = []
    errors = []
    windowed = []
    completed_since_save = 0
    sequences = list(SeqIO.parse(str(fasta_path), "fasta"))
    for index, record in enumerate(sequences, start=1):
        pdb_id = record.id.split("|")[0]
        if pdb_id == "5ies_0P" or pdb_id in payload["esm2_650m"]:
            continue
        parts = str(record.seq).split(":")
        graph_path = graph_dir / f"{pdb_id}.pt"
        if len(parts) < 3 or not graph_path.exists():
            skipped.append({"pdb_id": pdb_id, "reason": "invalid_fasta_or_missing_graph"})
            continue
        antigen_sequence = parts[2]
        try:
            embedding, window_count = embed_sequence(model, batch_converter, antigen_sequence, device)
            if window_count > 1:
                windowed.append({"pdb_id": pdb_id, "length": len(antigen_sequence), "windows": window_count})
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            surface_mask = torch.as_tensor(graph["mapping"]["ag"]["seqres2surf"], dtype=torch.bool)
            if embedding.shape[0] != surface_mask.shape[0]:
                raise ValueError(f"embedding/mask length mismatch: {embedding.shape[0]} != {surface_mask.shape[0]}")
            payload["esm2_650m"][pdb_id] = embedding[surface_mask].numpy()
            completed_since_save += 1
            if completed_since_save >= args.checkpoint_every:
                atomic_torch_save(payload, output_path)
                completed_since_save = 0
                print(f"saved={len(payload['esm2_650m'])}/{len(sequences)}", flush=True)
        except Exception as exc:
            errors.append({"pdb_id": pdb_id, "reason": repr(exc)})
            print(f"ERROR {pdb_id}: {exc!r}", flush=True)

    atomic_torch_save(payload, output_path)
    torch.cuda.synchronize()
    report = {
        "status": "complete" if not errors else "complete_with_errors",
        "model": "esm2_t33_650M_UR50D",
        "device": torch.cuda.get_device_name(0),
        "input_records": len(sequences),
        "embedded_complexes": len(payload["esm2_650m"]),
        "skipped": skipped,
        "errors": errors,
        "windowed_complexes": windowed,
        "windowing_method": "1022-residue windows, 128-residue minimum overlap, mean in overlap",
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
