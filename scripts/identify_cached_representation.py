#!/usr/bin/env python3
"""Identify the AsEP 'igfold' cache by running AntiBERTy on the full antibody.

The first attempt fed AntiBERTy the CDR-only sequence the manifest carries, 60
residues where the antibody is 224. AntiBERTy is contextual, so that compares
AntiBERTy-of-a-fragment against a cache that -- if it is AntiBERTy at all -- is
AntiBERTy-of-the-domain, selected afterwards. Nothing follows from a mismatch
there.

This uses `ab_seq_full`. The heavy/light boundary is not recorded, and AntiBERTy
encodes chains separately, so the boundary is swept over the range a VH domain
plausibly occupies and the best match is reported. Every cached row is matched
to its nearest row among the fresh ones, which needs no knowledge of the CDR
mask: if the cache is AntiBERTy of this domain, each of its rows must appear
among AntiBERTy's rows for the same domain.

A collapse to near-zero at some split identifies the cache. A floor that stays
near the vector norm at every split leaves it unidentified -- not refuted, since
a layer or version difference would also fail to match.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def load_cached(root: Path, key: str):
    path = root / key[:2] / f"{key}.pt"
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return torch.cat([torch.as_tensor(c).float() for c in payload["chunks"]], dim=0)


def main() -> int:
    import pandas

    from antiberty import AntiBERTyRunner

    instances = Path(sys.argv[1])
    root = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    runner = AntiBERTyRunner()
    frame = pandas.read_parquet(instances)

    checked = 0
    for _, row in frame.iterrows():
        cached = load_cached(root, str(row["ab_sequence_sha256"]))
        if cached is None:
            continue
        full = str(row["ab_seq_full"])
        norm = float(cached.norm(dim=1).mean())

        best = None
        for split in list(range(100, 141)) + [len(full)]:
            if split >= len(full):
                chains = [full]
            else:
                chains = [full[:split], full[split:]]
            try:
                embeddings = runner.embed(chains)
            except Exception:
                continue
            fresh = torch.cat([e[1:-1].float() for e in embeddings], dim=0)
            if fresh.shape[0] < cached.shape[0]:
                continue
            distance = float(torch.cdist(cached, fresh).min(dim=1).values.mean())
            if best is None or distance < best[0]:
                best = (distance, split, fresh.shape[0])

        if best is None:
            continue
        distance, split, rows = best
        print(f"{row['instance_id']}: cached {tuple(cached.shape)}  full {len(full)}aa")
        print(f"   best split at {split} (fresh rows {rows})")
        print(f"   mean nearest-neighbour distance {distance:.6f}   "
              f"cached row norm {norm:.4f}   ratio {distance / norm:.6f}")
        checked += 1
        if checked >= limit:
            break

    if checked == 0:
        print("no instance had both a full sequence and a cached tensor", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
