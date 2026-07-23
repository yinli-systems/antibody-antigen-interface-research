#!/usr/bin/env python3
"""Build only the ESM-2 650M residue graph tensor used by EpiFormer."""

import argparse
import gc
import os
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch


_WORKER_CREATOR_CLASS = None
_WORKER_CONFIG = None
_WORKER_ASEP_GRAPHS = None
_WORKER_ANTIBERTY = None
_WORKER_AG_PLM = None
_WORKER_CREATOR = None
_WORKER_SHARD_DIR = None


def _initialize_graph_worker():
    global _WORKER_CREATOR
    torch.set_num_threads(1)
    _WORKER_CREATOR = _WORKER_CREATOR_CLASS(
        **_WORKER_CONFIG,
        ag_plm_embeddings=_WORKER_AG_PLM,
    )


def _process_graph_worker(pdb_id):
    try:
        output_path = Path(_WORKER_SHARD_DIR) / f"{pdb_id}.pt"
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        graph = _WORKER_CREATOR.process_complex(
            pdb_id, _WORKER_ASEP_GRAPHS, _WORKER_ANTIBERTY
        )
        torch.save(graph, temporary_path)
        temporary_path.replace(output_path)
        return pdb_id, None
    except Exception as error:
        return pdb_id, repr(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    sys.path.insert(0, str(source_root / "code" / "data"))
    from construct_res_graphs_tensor import HierarchicalDatasetCreator

    class CorrectedSplitDatasetCreator(HierarchicalDatasetCreator):
        """Bridge upstream split and overly frequent checkpointing issues.

        The corrected AsEP preprocessing output has already removed 5nj6_0P and
        5ies_0P, while ``create_dataset`` unconditionally removes both again.
        The upstream builder also serializes the complete, growing dataset every
        ten complexes. This implementation uses idempotent filtering and atomic
        checkpoints every 250 new complexes while preserving resume behavior.
        """

        def _atomic_save(self, payload):
            output_path = Path(self.output_path)
            temporary_path = output_path.with_name(output_path.name + ".tmp")
            torch.save(payload, temporary_path)
            temporary_path.replace(output_path)

        def create_dataset(self, max_examples=None):
            print("Creating hierarchical dataset (corrected resume mode)...", flush=True)
            asep_graphs = self.load_asep_data()
            antiberty_embeddings = torch.load(
                self.antiberty_path, map_location="cpu", weights_only=False
            )

            bad_ids = {"5nj6_0P", "5ies_0P"}
            self.compatibility_skipped = sorted(bad_ids.intersection(asep_graphs))
            pdb_ids = [pdb_id for pdb_id in asep_graphs if pdb_id not in bad_ids]

            output_path = Path(self.output_path)
            shard_dir = output_path.parent / "res_graph_tensor_esm2_650m_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)

            # Convert any earlier monolithic checkpoint into per-complex shards.
            if output_path.exists():
                try:
                    processed_data = torch.load(
                        output_path, map_location="cpu", weights_only=False
                    )
                    for data in processed_data:
                        shard_path = shard_dir / f"{data.complex_id}.pt"
                        if not shard_path.exists():
                            temporary_path = shard_path.with_name(shard_path.name + ".tmp")
                            torch.save(data, temporary_path)
                            temporary_path.replace(shard_path)
                    print(
                        f"Imported {len(processed_data)} monolithic checkpoint complexes",
                        flush=True,
                    )
                    del processed_data
                    gc.collect()
                except Exception as error:
                    print(
                        f"Existing monolithic checkpoint is unreadable; using shards: {error}",
                        flush=True,
                    )

            existing_ids = {path.stem for path in shard_dir.glob("*.pt")}
            remaining_ids = [pdb_id for pdb_id in pdb_ids if pdb_id not in existing_ids]
            if max_examples:
                remaining_ids = remaining_ids[:max_examples]
            print(f"Remaining complexes: {len(remaining_ids)}", flush=True)

            worker_count = max(1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
            torch.set_num_threads(1)
            print(f"graph_workers={worker_count}", flush=True)

            global _WORKER_CREATOR_CLASS, _WORKER_CONFIG, _WORKER_SHARD_DIR
            global _WORKER_ASEP_GRAPHS, _WORKER_ANTIBERTY, _WORKER_AG_PLM
            _WORKER_CREATOR_CLASS = HierarchicalDatasetCreator
            _WORKER_CONFIG = {
                "asep_data_path": self.asep_data_path,
                "ag_pdb_dir": self.ag_pdb_dir,
                "ab_pdb_dir": self.ab_pdb_dir,
                "output_path": self.output_path,
                "antiberty_path": self.antiberty_path,
            }
            _WORKER_ASEP_GRAPHS = asep_graphs
            _WORKER_ANTIBERTY = antiberty_embeddings
            _WORKER_AG_PLM = self.ag_plm_embeddings
            _WORKER_SHARD_DIR = str(shard_dir)

            errors = []
            context = mp.get_context("fork")
            with context.Pool(
                processes=worker_count,
                initializer=_initialize_graph_worker,
            ) as pool:
                results = pool.imap_unordered(
                    _process_graph_worker, remaining_ids, chunksize=1
                )
                for index, (pdb_id, error) in enumerate(results, start=1):
                    if error is not None:
                        errors.append({"pdb_id": pdb_id, "error": error})
                    if index % 25 == 0 or index == len(remaining_ids):
                        print(
                            f"processed_new={index}/{len(remaining_ids)} "
                            f"errors={len(errors)}",
                            flush=True,
                        )

            self.errors = errors
            if errors:
                raise RuntimeError(f"Graph construction failed for {len(errors)} complexes")

            missing = [pdb_id for pdb_id in pdb_ids if not (shard_dir / f"{pdb_id}.pt").exists()]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} graph shards: {missing[:10]}")

            print("Merging graph shards into final dataset...", flush=True)
            processed_data = [
                torch.load(shard_dir / f"{pdb_id}.pt", map_location="cpu", weights_only=False)
                for pdb_id in pdb_ids
            ]
            self._atomic_save(processed_data)
            self.shard_count = len(processed_data)
            return processed_data

    esm_path = data_root / "antigen" / "plm_embeddings" / "ag_esm2_embeddings_asep.pt"
    antiberty_path = data_root / "antibody" / "antiberty_embeddings" / "asep_antiberty_embeddings.pt"
    output_path = data_root / "m3epi" / "res_graph_tensor_esm2_650m.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    esm_payload = torch.load(esm_path, map_location="cpu", weights_only=False)
    creator = CorrectedSplitDatasetCreator(
        asep_data_path=str(data_root / "processed" / "dict_pre_cal.pt"),
        ag_pdb_dir=str(data_root / "antigen" / "atmseq2surf"),
        ab_pdb_dir=str(data_root / "antibody" / "atmseq2cdr"),
        antiberty_path=str(antiberty_path),
        output_path=str(output_path),
        ag_plm_embeddings=esm_payload["esm2_650m"],
    )
    dataset = creator.create_dataset()
    report = {
        "status": "complete",
        "complexes": len(dataset),
        "elapsed_seconds": time.time() - started,
        "output": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "upstream_compatibility_fix": {
            "issue": "corrected preprocessing already removed 5nj6_0P and 5ies_0P",
            "bad_ids_present_and_skipped": creator.compatibility_skipped,
            "checkpoint_strategy": "one atomic shard per complex, then atomic final merge",
            "graph_shards": creator.shard_count,
            "graph_workers": max(
                1, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            ),
        },
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
