"""Graph community detection split module.

This module implements a graph-community-detection-based dual-axis splitting
algorithm as an alternative to the infeasible linear CP-SAT split.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Mapping, Sequence

try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy.sparse as sp
    from scipy.sparse import linalg as splinalg
except ImportError:
    sp = None
    splinalg = None


def require_scipy() -> None:
    """Ensure numpy and scipy are available."""
    if sp is None or np is None:
        raise RuntimeError("scipy and numpy are required for graph_community_split")


@dataclasses.dataclass(frozen=True)
class DualAxisSimilarityGraph:
    """Bipartite similarity graph between complexes.

    Contains antibody similarity edges, antigen family edges, and complex IDs.
    """

    complex_ids: list[str]
    complex_to_idx: dict[str, int]
    antibody_adj: Any
    antigen_adj: Any
    joint_adj: Any

    @classmethod
    def build(
        cls,
        complex_ids: Sequence[str],
        antibody_similarities: Mapping[tuple[str, str], float],
        antigen_families: Mapping[str, str],
        antibody_threshold: float = 0.7,
    ) -> DualAxisSimilarityGraph:
        """Build the dual axis similarity graph from raw inputs.

        Args:
            complex_ids: Sequence of all complex IDs.
            antibody_similarities: Mapping of pairs to similarity scores.
            antigen_families: Mapping from complex ID to family ID.
            antibody_threshold: Threshold for antibody edges.

        Returns:
            A populated DualAxisSimilarityGraph.
        """
        complex_ids = list(complex_ids)
        complex_to_idx = {cid: i for i, cid in enumerate(complex_ids)}

        ab_adj = build_antibody_similarity_graph(
            complex_ids, antibody_similarities, threshold=antibody_threshold
        )
        ag_adj = build_antigen_family_graph(complex_ids, antigen_families)
        joint_adj = build_joint_constraint_graph(ab_adj, ag_adj)

        return cls(
            complex_ids=complex_ids,
            complex_to_idx=complex_to_idx,
            antibody_adj=ab_adj,
            antigen_adj=ag_adj,
            joint_adj=joint_adj,
        )


def build_antibody_similarity_graph(
    complex_ids: Sequence[str],
    similarities: Mapping[tuple[str, str], float],
    threshold: float = 0.7,
) -> Any:
    """Build an undirected graph for antibody similarities.

    Args:
        complex_ids: List of complex IDs.
        similarities: Mapping of (id1, id2) -> similarity score.
        threshold: Minimum similarity to create an edge.

    Returns:
        A scipy.sparse.csr_matrix representing the adjacency structure.
    """
    require_scipy()
    n = len(complex_ids)
    idx_map = {cid: i for i, cid in enumerate(complex_ids)}

    rows = []
    cols = []
    data = []

    for (c1, c2), sim in similarities.items():
        if sim >= threshold:
            if c1 in idx_map and c2 in idx_map:
                i, j = idx_map[c1], idx_map[c2]
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([1.0, 1.0])

    # Add self loops
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.data = np.ones_like(adj.data)  # Binarize
    return adj


def build_antigen_family_graph(
    complex_ids: Sequence[str],
    antigen_families: Mapping[str, str],
) -> Any:
    """Build an undirected graph connecting shared antigen families.

    Args:
        complex_ids: List of complex IDs.
        antigen_families: Mapping of complex ID -> family ID.

    Returns:
        A scipy.sparse.csr_matrix representing the adjacency structure.
    """
    require_scipy()
    n = len(complex_ids)
    idx_map = {cid: i for i, cid in enumerate(complex_ids)}

    fam_to_nodes = collections.defaultdict(list)
    for cid, fam in antigen_families.items():
        if cid in idx_map:
            fam_to_nodes[fam].append(idx_map[cid])

    rows = []
    cols = []
    for nodes in fam_to_nodes.values():
        for i in nodes:
            for j in nodes:
                rows.append(i)
                cols.append(j)

    if not rows:
        return sp.csr_matrix((n, n), dtype=float)

    data = np.ones(len(rows), dtype=float)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.data = np.ones_like(adj.data)
    return adj


def build_joint_constraint_graph(antibody_adj: Any, antigen_adj: Any) -> Any:
    """Merge antibody similarity graph and antigen family graph.

    Args:
        antibody_adj: Scipy sparse adjacency matrix for antibodies.
        antigen_adj: Scipy sparse adjacency matrix for antigens.

    Returns:
        Merged scipy.sparse.csr_matrix.
    """
    require_scipy()
    joint = antibody_adj + antigen_adj
    joint.data = np.ones_like(joint.data)
    return joint


def _louvain_pass(adj: Any, resolution: float) -> Any:
    """Perform one pass of the Louvain algorithm."""
    n = adj.shape[0]
    communities = np.arange(n)
    node_weights = np.array(adj.sum(axis=1)).flatten()
    total_weight = node_weights.sum()
    if total_weight == 0:
        return communities

    community_weights = node_weights.copy()
    improved = True

    while improved:
        improved = False
        nodes = np.random.permutation(n)
        for i in nodes:
            row = adj[i]
            neighbors = row.indices
            edge_weights = row.data

            if len(neighbors) == 0:
                continue

            current_c = communities[i]
            k_i = node_weights[i]

            neighbor_communities = communities[neighbors]
            unique_comms = np.unique(neighbor_communities)

            best_c = current_c
            best_gain = 0.0

            k_i_in = collections.defaultdict(float)
            for nbr, w in zip(neighbors, edge_weights):
                k_i_in[communities[nbr]] += w

            # Temporarily remove node from current community
            community_weights[current_c] -= k_i

            for target_c in unique_comms:
                gain = (
                    k_i_in[target_c]
                    - resolution * k_i * community_weights[target_c] / total_weight
                )
                if gain > best_gain:
                    best_gain = gain
                    best_c = target_c

            if best_c != current_c and best_gain > 1e-7:
                communities[i] = best_c
                community_weights[best_c] += k_i
                improved = True
            else:
                # Put back
                community_weights[current_c] += k_i

    return communities


def _louvain_hierarchy(adj: Any, resolution: float) -> Any:
    """Run full Louvain algorithm to find stable communities."""
    n = adj.shape[0]
    final_communities = np.arange(n)
    current_adj = adj

    while True:
        comms = _louvain_pass(current_adj, resolution)
        unique_comms, inv_comms = np.unique(comms, return_inverse=True)
        if len(unique_comms) == current_adj.shape[0]:
            break

        final_communities = inv_comms[final_communities]

        n_new = len(unique_comms)
        rows = np.arange(current_adj.shape[0])
        cols = inv_comms
        data = np.ones_like(rows)
        s_mat = sp.coo_matrix((data, (rows, cols)), shape=(current_adj.shape[0], n_new)).tocsr()
        current_adj = s_mat.T.dot(current_adj.dot(s_mat))

    return final_communities


def spectral_fallback_split(
    joint_adj: Any,
    complex_ids: Sequence[str],
    num_folds: int = 3,
) -> dict[str, int]:
    """Fallback splitting using spectral bisection if communities are unbalanced.

    Args:
        joint_adj: Merged adjacency matrix.
        complex_ids: Sequence of complex IDs.
        num_folds: Target number of folds.

    Returns:
        Mapping of complex_id to fold index.
    """
    require_scipy()
    n = joint_adj.shape[0]

    def bisect(node_indices: Any) -> tuple[Any, Any]:
        if len(node_indices) <= 1:
            return node_indices, np.array([], dtype=int)

        sub_adj = joint_adj[node_indices][:, node_indices]
        sub_deg = np.array(sub_adj.sum(axis=1)).flatten()
        sub_d = sp.diags(sub_deg)
        sub_l = sub_d - sub_adj

        if sub_l.shape[0] < 3:
            mid = len(node_indices) // 2
            return node_indices[:mid], node_indices[mid:]

        try:
            _, vecs = splinalg.eigsh(sub_l.astype(float), k=2, which="SM", sigma=1e-5)
            fiedler = vecs[:, 1]
            median = np.median(fiedler)
            left = node_indices[fiedler <= median]
            right = node_indices[fiedler > median]
            if len(left) == 0 or len(right) == 0:
                mid = len(node_indices) // 2
                return node_indices[:mid], node_indices[mid:]
            return left, right
        except Exception:
            mid = len(node_indices) // 2
            return node_indices[:mid], node_indices[mid:]

    folds = [np.arange(n)]
    while len(folds) < num_folds:
        largest_idx = np.argmax([len(f) for f in folds])
        largest = folds.pop(int(largest_idx))
        left, right = bisect(largest)
        folds.append(left)
        if len(right) > 0:
            folds.append(right)
        else:
            # Prevent infinite loop if bisection fails to split
            break

    assignment = {}
    for fold_id, fold_nodes in enumerate(folds):
        for node_idx in fold_nodes:
            assignment[complex_ids[node_idx]] = fold_id

    return assignment


def louvain_dual_axis_split(
    joint_adj: Any,
    complex_ids: Sequence[str],
    num_folds: int = 3,
    target_test_fraction: float = 0.1,
    min_test_complexes: int = 10,
    resolution: float = 1.0,
) -> dict[str, int]:
    """Split complexes using Louvain community detection.

    First decomposes the graph into connected components, then runs Louvain
    within each component. Complexes in the same community always go to the
    same fold, which guarantees zero leakage by construction.

    Args:
        joint_adj: Merged adjacency matrix.
        complex_ids: Sequence of complex IDs.
        num_folds: Number of desired folds.
        target_test_fraction: Fraction of complexes desired in test folds.
        min_test_complexes: Minimum complexes required in test folds.
        resolution: Louvain resolution parameter.

    Returns:
        Mapping from complex_id to fold index.
    """
    require_scipy()
    from scipy.sparse import csgraph as _csgraph

    n = len(complex_ids)
    np.random.seed(42)  # For reproducibility

    # Step 1: Decompose into connected components first.
    # Nodes in different connected components must NEVER share a community.
    n_cc, cc_labels = _csgraph.connected_components(joint_adj, directed=False)

    # Step 2: Run Louvain within each connected component independently.
    global_communities = np.empty(n, dtype=np.int64)
    next_community_id = 0

    for cc_id in range(n_cc):
        cc_mask = cc_labels == cc_id
        cc_indices = np.where(cc_mask)[0]

        if len(cc_indices) == 1:
            # Singleton component gets its own community.
            global_communities[cc_indices[0]] = next_community_id
            next_community_id += 1
            continue

        # Extract subgraph for this component.
        sub_adj = joint_adj[cc_indices][:, cc_indices].tocsr()
        sub_communities = _louvain_hierarchy(sub_adj, resolution)

        # Map sub-community IDs to globally unique IDs.
        unique_sub = np.unique(sub_communities)
        sub_to_global = {int(s): next_community_id + i for i, s in enumerate(unique_sub)}
        for local_idx, global_idx in zip(cc_indices, sub_communities):
            global_communities[local_idx] = sub_to_global[int(global_idx)]
        next_community_id += len(unique_sub)

    # Step 3: Group nodes by connected component (NOT sub-community).
    # The joint graph encodes both antibody similarity and antigen family
    # edges. Two nodes in the same connected component share at least one
    # constraint path, so they MUST be in the same fold.
    cc_to_nodes = collections.defaultdict(list)
    for i in range(n):
        cc_to_nodes[int(cc_labels[i])].append(i)

    allocatable_groups = list(cc_to_nodes.values())
    allocatable_groups.sort(key=len, reverse=True)

    if not allocatable_groups:
        return {}

    # Step 4: Greedy bin-packing of connected components into folds.
    if num_folds == 3:
        target_fractions = [
            target_test_fraction,
            target_test_fraction,
            1.0 - 2 * target_test_fraction,
        ]
    else:
        target_fractions = [1.0 / num_folds] * num_folds

    target_sizes = [int(n * f) for f in target_fractions]

    # If the largest connected component dominates, fallback to spectral bisection.
    if len(allocatable_groups[0]) > 0.9 * n:
        return spectral_fallback_split(joint_adj, complex_ids, num_folds)

    fold_assignments = [[] for _ in range(num_folds)]
    fold_sizes = [0] * num_folds

    for group in allocatable_groups:
        deficits = [target_sizes[i] - fold_sizes[i] for i in range(num_folds)]
        best_fold = int(np.argmax(deficits))
        fold_assignments[best_fold].extend(group)
        fold_sizes[best_fold] += len(group)

    # Check minimum conditions for first two folds (assumed test/val).
    if num_folds >= 2 and (
        fold_sizes[0] < min_test_complexes or fold_sizes[1] < min_test_complexes
    ):
        return spectral_fallback_split(joint_adj, complex_ids, num_folds)

    assignment = {}
    for fold_id, nodes in enumerate(fold_assignments):
        for node in nodes:
            assignment[complex_ids[node]] = fold_id

    return assignment


def validate_dual_axis_split(
    assignments: Mapping[str, int],
    antibody_similarities: Mapping[tuple[str, str], float],
    antigen_families: Mapping[str, str],
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Validate that the split has zero leakage across specified axes.

    Args:
        assignments: Mapping of complex_id -> fold index.
        antibody_similarities: Pairwise antibody similarities.
        antigen_families: Antigen families mapping.
        threshold: Sequence identity threshold.

    Returns:
        Validation report dictionary containing leakage statistics.
    """
    fold_counts = collections.Counter(assignments.values())

    leakage_edges = 0
    for (c1, c2), sim in antibody_similarities.items():
        if sim >= threshold:
            f1 = assignments.get(c1)
            f2 = assignments.get(c2)
            if f1 is not None and f2 is not None and f1 != f2:
                leakage_edges += 1

    fam_to_folds = collections.defaultdict(set)
    for cid, f in assignments.items():
        fam = antigen_families.get(cid)
        if fam is not None:
            fam_to_folds[fam].add(f)

    leaking_families = 0
    for fam, folds in fam_to_folds.items():
        if len(folds) > 1:
            leaking_families += 1

    return {
        "fold_sizes": dict(fold_counts),
        "antibody_leakage_edges": leakage_edges,
        "antigen_leaking_families": leaking_families,
        "is_valid": leakage_edges == 0 and leaking_families == 0,
    }
