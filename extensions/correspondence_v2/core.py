"""Label-free perturbations and paired statistics; no historical files modified."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence
import numpy as np

MODES = ("identity", "chain_shift", "local_shuffle", "same_aa_shuffle",
         "global_shuffle", "oracle_restore")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    """Exclusive creation: a failed or previous result must not be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def chain_permutation(sequences: Sequence[str], mode: str, severity: int,
                      seed: int, key: str) -> tuple[np.ndarray, dict]:
    """Return destination->source row mapping, never crossing a biological chain.

    Circular shifts preserve every feature vector but have boundary wraparound;
    linear displacement and wrap counts are explicitly reported. The oracle
    restore is a plumbing check, NOT a learned rescue algorithm.
    """
    if mode not in MODES:
        raise ValueError(f"unknown perturbation {mode!r}")
    if not sequences or any(not isinstance(s, str) or not s or not s.isascii()
                            for s in sequences):
        raise ValueError("nonempty ASCII sequences required for every chain")
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError("severity must be an integer")
    if mode == "local_shuffle" and severity < 1:
        raise ValueError("local_shuffle block size must be positive")
    rng_seed = int.from_bytes(hashlib.sha256(
        f"{seed}:{key}:{mode}:{severity}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    n = sum(map(len, sequences))
    order = np.arange(n, dtype=np.int64)
    aa = np.array(list("".join(sequences)))
    chain_ids = np.concatenate([np.full(len(s), i) for i, s in enumerate(sequences)])
    start, wraps = 0, 0
    for seq in sequences:
        length = len(seq)
        ids = np.arange(start, start + length)
        if mode in ("chain_shift", "oracle_restore"):
            order[ids] = start + (np.arange(length) + severity) % length
            wraps += int(np.sum((np.arange(length) + severity < 0) |
                                (np.arange(length) + severity >= length)))
        elif mode == "global_shuffle":
            order[ids] = rng.permutation(ids)
        elif mode == "same_aa_shuffle":
            for symbol in sorted(set(seq)):
                group = ids[np.array(list(seq)) == symbol]
                order[group] = rng.permutation(group)
        elif mode == "local_shuffle":
            for offset in range(0, length, severity):
                block = ids[offset:offset + severity]
                order[block] = rng.permutation(block)
        start += length
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise AssertionError("mapping is not bijective")
    if np.any(chain_ids != chain_ids[order]):
        raise AssertionError("biological chain boundary crossed")
    report = {
        "mode": mode, "severity": severity, "length": n,
        "changed_fraction": float(np.mean(order != np.arange(n))),
        "linear_displacement_mean": float(np.abs(order - np.arange(n)).mean()),
        "linear_displacement_max": int(np.abs(order - np.arange(n)).max()),
        "amino_acid_changed_fraction": float(np.mean(aa != aa[order])),
        "chain_crossings": 0, "wraparound_rows": wraps,
        "order_sha256": hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
    }
    if mode == "oracle_restore":
        inverse = np.argsort(order)
        order = order[inverse]
        report["oracle_composition_is_identity"] = bool(np.array_equal(order, np.arange(n)))
    return order, report


def transform_extra_columns(tensor, base_dim: int, extra_dim: int,
                            order: np.ndarray):
    """Reorder extra features only; preserve ESM-2 and presence bits bitwise."""
    import torch
    if tensor.ndim != 2 or tensor.shape[1] != base_dim + extra_dim + 1:
        raise ValueError("expected base features + one extra source + presence bit")
    if len(order) != tensor.shape[0] or not np.array_equal(
            np.sort(order), np.arange(len(order))):
        raise ValueError("order must be a full residue permutation")
    if not torch.isfinite(tensor).all():
        raise ValueError("non-finite input features")
    presence = tensor[:, -1]
    if not bool(((presence == 0) | (presence == 1)).all()):
        raise ValueError("presence bit is not binary")
    # Partial availability would move features across missingness; fail closed.
    if not bool((presence == presence[0]).all()):
        raise ValueError("partial-residue missingness needs a separate protocol")
    out = tensor.clone()
    idx = torch.as_tensor(order.copy(), device=tensor.device, dtype=torch.long)
    out[:, base_dim:base_dim + extra_dim] = tensor[:, base_dim:base_dim + extra_dim].index_select(0, idx)
    if not torch.equal(out[:, :base_dim], tensor[:, :base_dim]):
        raise AssertionError("base features changed")
    if not torch.equal(out[:, -1], tensor[:, -1]):
        raise AssertionError("presence mask changed")
    return out


def ranking_summary(scores: Sequence[float], positive: Sequence[bool]) -> dict:
    """Average tie handling; supports multiple known positives.

    Unknown partners are retrieval distractors, not demonstrated non-binders.
    Reciprocal rank is the expectation for the first positive under uniformly
    random ordering within the first positive tie block.
    """
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    if scores.ndim != 1 or scores.shape != positive.shape or len(scores) < 2:
        raise ValueError("matching one-dimensional candidate arrays required")
    if not np.isfinite(scores).all() or not positive.any() or positive.all():
        raise ValueError("finite scores and positive/distractor candidates required")
    best = float(scores[positive].max())
    before = int(np.sum(scores > best))
    tied = scores == best
    size, pos = int(tied.sum()), int(positive[tied].sum())
    # Distribution of earliest success when selecting pos slots out of size.
    weights = [math.comb(size-j, pos-1) / math.comb(size, pos)
               for j in range(1, size-pos+2)]
    mrr = sum(w / (before+j) for j, w in enumerate(weights, 1))
    recall1 = pos / size if before == 0 else 0.0
    return {"mrr": float(mrr), "recall_at_1": float(recall1),
            "positive_count": int(positive.sum()), "candidate_count": len(scores),
            "known_positive_minus_distractor_mean": float(scores[positive].mean()-scores[~positive].mean()),
            "tie_policy": "uniform_random_within_exact_score_ties"}


def crossed_cluster_bootstrap(differences: np.ndarray, clusters: Sequence[str],
                              seed: int = 20260930, repeats: int = 5000) -> dict:
    """Paired seed x complex differences; resample seeds and whole clusters.

    The primary estimand gives equal weight to clusters, then equal weight to
    complexes within each cluster. The row/complex-weighted mean is separately
    reported. This is development uncertainty, not an independent-test CI.
    """
    x = np.asarray(differences, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(clusters) or min(x.shape) < 1:
        raise ValueError("expected nonempty seed x complex matrix")
    if not np.isfinite(x).all() or repeats < 100:
        raise ValueError("finite complete pairs and >=100 bootstrap draws required")
    names = sorted(set(map(str, clusters)))
    if len(names) < 2:
        raise ValueError("at least two independent clusters required")
    c = np.asarray(list(map(str, clusters)))
    matrix = np.stack([x[:, c == name].mean(axis=1) for name in names], axis=1)
    rng = np.random.default_rng(seed)
    estimates, conditional = [], []
    for _ in range(repeats):
        cs = rng.integers(len(names), size=len(names))
        ss = rng.integers(x.shape[0], size=x.shape[0])
        estimates.append(matrix[ss][:, cs].mean())
        conditional.append(matrix[:, cs].mean())
    return {
        "estimand": "equal_cluster_equal_seed_mean_paired_difference",
        "mean_difference": float(matrix.mean()),
        "complex_weighted_mean_difference": float(x.mean()),
        "crossed_seed_cluster_ci95": list(map(float, np.quantile(estimates, [.025,.975]))),
        "cluster_ci95_conditional_on_fixed_seeds": list(map(float,np.quantile(conditional,[.025,.975]))),
        "seed_count": x.shape[0], "complex_count": x.shape[1],
        "cluster_count": len(names), "bootstrap_repeats": repeats,
        "positive_cluster_fraction": float(np.mean(matrix.mean(axis=0)>0)),
        "negative_cluster_fraction": float(np.mean(matrix.mean(axis=0)<0)),
        "scope": "exploratory development; pointwise intervals, no promotion decision",
    }


def guard_validation_ids(ids: Sequence[str], splits: dict) -> None:
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique evaluation IDs required")
    validation = set(map(str, splits.get("validation", [])))
    forbidden = set(map(str, splits.get("train", []))) | set(map(str, splits.get("test", [])))
    if not set(ids) <= validation or set(ids) & forbidden:
        raise ValueError("test firewall: only disjoint development-validation IDs allowed")
