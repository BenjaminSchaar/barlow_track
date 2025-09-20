"""
HDBSCAN condensed-tree agglomeration with strict time-purity metric.

Inputs (provided by you / earlier pipeline):
- clusterer: a fitted hdbscan.HDBSCAN object (after clusterer.fit(X) )
- linear_ind_to_t_and_seg_id: dict mapping linear_index -> (t, seg_id, other_id)
    - only the 't' is used here; linear_index is the row index in your original
      embedding / feature matrix
- time_index_to_linear_feature_indices: dict mapping t -> list[linear_index]
    - inverse of the above mapping
- (optional) min_kept_size: minimum number of points required for an accepted cluster

Outputs:
- accepted_clusters: list of dicts describing each accepted merged cluster:
    {'node_id': <condensed-tree node id>,
     'raw_indices': set(...),         # raw indices (all leaves under node)
     'kept_indices': set(...),        # indices kept after enforcing strict one-per-time
     'purity': float }                # purity = len(kept_indices) / len(raw_indices)
- assigned_index_to_cluster: dict mapping linear_index -> accepted_cluster_idx
- outliers: set of linear indices marked as outliers because they lost same-time collisions


Example usage:
accepted_clusters, assigned_map, outliers = agglomerate_by_time_purity(
    clusterer=clusterer,
    linear_ind_to_t_and_seg_id=linear_ind_to_t_and_seg_id,
    time_index_to_linear_feature_indices=time_index_to_linear_feature_indices,
    min_kept_size=2
)

After running you can inspect:
- how many clusters were accepted: len(accepted_clusters)
- distribution of purity: [c['purity'] for c in accepted_clusters]
- unassigned points (neither assigned nor outlier) = set(range(n_points)) - assigned_map.keys() - outliers
"""

from typing import Dict, List, Tuple, Set, Iterable, Optional
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm.auto import tqdm
import random


def compute_time_purity_for_indices(indices: Iterable[int],
                                    linear_ind_to_t_and_seg_id: Dict[int, Tuple],
                                    clusterer,
                                    assigned_indices: Optional[Set[int]] = None,
                                   max_size=None):
    """
    For a set of indices (raw cluster candidate), compute:
      - kept_indices: enforce strict one-per-time by selecting the single highest-confidence
                      index for each timepoint
      - purity = len(kept_indices) / len(indices)  (float in [0,1])
      - conflicted_times: list of timepoints that had >1 candidate
    Parameters:
      - indices: iterable of linear indices (point-level)
      - linear_ind_to_t_and_seg_id: mapping from index -> (t, seg_id, other_id)
      - clusterer: to read per-point probabilities_ (fallback behavior described below)
      - assigned_indices: optional set of already-assigned indices; if provided,
          we will *prefer* unassigned candidates when choosing between same-time points.
          (If the best-confidence point is already assigned elsewhere, this routine will
           still return it as 'best' — the caller should check assigned conflicts and
           decide whether to accept this candidate.)
    Returns:
      kept_indices (set), purity (float), map_time_to_candidates (dict)
    """
    indices = set(indices)
    if len(indices) == 0:
        return set(), 0.0, {}

    # Try to obtain membership confidence/probability for each point.
    # HDBSCAN provides clusterer.probabilities_ (0 for noise, (0,1] for cluster members).
    if hasattr(clusterer, "probabilities_") and clusterer.probabilities_ is not None:
        probs = getattr(clusterer, "probabilities_")
        # guard length
        if len(probs) < max(indices) + 1:
            # fallback to uniform if mismatched sizes
            def _score(i): return 1.0
        else:
            def _score(i): return float(probs[int(i)])
    else:
        # fallback: use presence in a cluster (labels_ != -1) as confidence 1.0,
        # noise (label -1) as 0.0. (This is crude — replace with embedding-based score if available.)
        labels = getattr(clusterer, "labels_", None)
        if labels is None:
            def _score(i): return 1.0
        else:
            def _score(i):
                lab = int(labels[int(i)])
                return 0.0 if lab == -1 else 1.0

    # group indices by time
    time_to_indices: Dict = {}
    for i in indices:
        if i not in linear_ind_to_t_and_seg_id:
            raise KeyError(f"Index {i} missing from linear_ind_to_t_and_seg_id mapping")
        t = linear_ind_to_t_and_seg_id[int(i)][0]
        time_to_indices.setdefault(t, []).append(i)

    kept_indices = set()
    conflicted_times = []
    for t, cand_list in time_to_indices.items():
        if len(cand_list) == 1:
            kept_indices.add(cand_list[0])
            continue
        # multiple candidates from the same timepoint: pick the highest score.
        # If assigned_indices provided, prefer unassigned candidates when scores tie or very close.
        cand_list_sorted = sorted(cand_list, key=lambda idx: (_score(idx), idx), reverse=True)
        # If best candidate already assigned AND there exists an unassigned candidate,
        # prefer the highest-scoring unassigned candidate (to reduce conflicts). This is optional.
        if assigned_indices:
            best = cand_list_sorted[0]
            if best in assigned_indices:
                # find best unassigned if present
                unassigned = [c for c in cand_list_sorted if c not in assigned_indices]
                if len(unassigned) > 0:
                    chosen = unassigned[0]
                else:
                    chosen = best  # all assigned; fall back to best
            else:
                chosen = best
        else:
            chosen = cand_list_sorted[0]
        kept_indices.add(chosen)
        conflicted_times.append(t)

    purity = float(len(kept_indices)) / float(len(indices)) if len(indices) > 0 else 0.0
    return kept_indices, purity, time_to_indices



def initialize_timepoint_seeds_with_prior(
    clusterer, time_index_to_linear_feature_indices, template_timepoint, cluster_label2node, t_max, assigned_index_to_cluster={}
):
    """
    Initialize seeds from a single template time point, skipping:
      - Noise points in HDBSCAN
      - Points already assigned to a cluster in a previous iteration

    Parameters
    ----------
    clusterer : fitted hdbscan.HDBSCAN
        The fitted HDBSCAN object with cluster labels.
    linear_ind_to_t_and_seg_id : dict[int, (t, seg_id, other_id)]
        Mapping from linear index to time and IDs.
    template_timepoint : int
        The time point to use as template for seeding.
    assigned_index_to_cluster : dict[int, any]
        Mapping from linear index to existing cluster (from prior merges).

    Returns
    -------
    dict[seed_id -> set[int]]
        Seeds to start agglomeration. Each seed corresponds to a unique HDBSCAN cluster.
    """
    labels = clusterer.labels_
    seeds = {}
    used_cluster_ids = set()

    linear_idx = time_index_to_linear_feature_indices[template_timepoint]
    num_leaves, num_hdbscan, num_maximal = 0, 0, 0

    for idx in tqdm(linear_idx, desc="Checking clusters of objects at this time point"):
        if idx in assigned_index_to_cluster:
            num_maximal += 1
            continue  # skip points already part of maximal clusters

        if labels[idx] > 0:            
            # Already has an hdbscan cluster
            start_node = cluster_label2node[labels[idx]]
            cluster_size = np.sum(clusterer.labels_ == labels[idx])
            if cluster_size > t_max:
                print(f"Found very large hdbscan cluster ({cluster_size}; label={labels[idx]}), trying to find better starting from leaf")
                start_node = idx
                num_leaves += 1
            else:
                num_hdbscan += 1
        else:
            # Is a leaf; the idx is the same in the graph
            start_node = idx
            num_leaves += 1

        # Precompute ancestor list (upward)
        ancestors = list(nx.ancestors(G, start_node))
        ancestors.append(start_node)  # include self
        seeds[idx] = {'start_node': start_node, 'ancestors': ancestors, 'original_label': labels[idx]}

    print(f"Found seeds at time {template_timepoint}: leaves: {num_leaves}; maximal: {num_maximal}; hdbscan: {num_hdbscan}")
    return seeds


def agglomerate_by_time_purity(clusterer,
                               G,
                               leaves_under,
                               cluster_label2node,
                               linear_ind_to_t_and_seg_id: Dict[int, Tuple],
                               time_index_to_linear_feature_indices: Dict[int, List[int]],
                               min_kept_size: int = 2,
                               eps_purity_increase: float = 1e-6,
                              min_purity=0.5):
    """
    Main function to run the greedy, time-seeded agglomeration over the HDBSCAN condensed tree.

    Strategy implemented:
    - Build condensed tree graph and map every node -> set of leaf point indices under it.
    - Iterate over timepoints (seed order: descending by #points in that timepoint).
    - For each point index at that timepoint:
        - Ascend the condensed tree from that point (point node id) and consider each ancestor node
          (candidate cluster = all leaves under that ancestor).
        - Compute time-purity for candidate cluster (kept indices after de-duplication).
        - Select the ancestor that yields the highest purity. Ties broken toward larger kept size.
        - Accept the candidate *only if*:
            * purity is strictly > current best purity for that specific seed (by eps_purity_increase), and
            * none of kept_indices are already assigned to an accepted cluster (we keep disjoint clusters).
          If any kept_index already assigned, the candidate is skipped to avoid index re-use.
        - Upon acceptance: add an accepted_cluster record; mark kept indices as assigned; mark
          the other raw indices in the candidate (those not in kept_indices) as outliers.
    - Return accepted clusters, assigned map, and outliers set.

    Notes:
    - This greedy approach is deterministic given deterministic traversal order and ties rules.
    - You can easily change the seeding order (e.g., based on within-timepoint cluster quality).
    """
    print("Generating networkx version of tree...")
    n_points = len(clusterer.labels_)

    # Get condensed tree networkx graph
    if G is None:
        G = clusterer.condensed_tree_.to_networkx()

    # Leaves are just nodes < n_points (point indices)
    if leaves_under is None:
        leaves_under = {}
        for node in G.nodes:
            # Collect all descendants + itself, filter to 0..n_points-1
            desc = nx.descendants(G, node) | {node}
            leaves_under[node] = {d for d in desc if 0 <= d < n_points}
    
    # Roots are nodes with no parent
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    # df_ct = build_condensed_tree_graph(clusterer)
    # G, leaves_under, roots = build_tree_and_leaf_index(df_ct, n_points)

    # For quickly finding ancestors of a point-node, we can use networkx.ancestors(G, node).
    # Note: a leaf node is the point index itself (0..n_points-1). We include the leaf's own node
    # by considering node + its ancestors.
    assigned_index_to_cluster: Dict[int, int] = {}
    accepted_clusters = []  # list of dicts
    outliers: Set[int] = set()

    # Order timepoints by descending number of points (you can change this ordering easily).
    
    timepoints = list(time_index_to_linear_feature_indices.keys())
    random.shuffle(timepoints)
    # timepoints = sorted(list(time_index_to_linear_feature_indices.keys()),
    #                     key=lambda t: len(time_index_to_linear_feature_indices[t]),
    #                     reverse=True)

    cluster_counter = 0

    # Small helper to test if any of given indices already assigned:
    def any_assigned(indices):
        return any((idx in assigned_index_to_cluster) for idx in indices)

    # Iterate seeds
    for t in tqdm(timepoints, desc="Iteratively clustering from time points"):
        num_clusters_changed = 0
        print("="*100)
        print(f"Clustering from starting time {t}")
        point_indices_for_t = list(time_index_to_linear_feature_indices[t])
        # Randomize or sort point order for reproducibility; we'll sort by index
        point_indices_for_t = sorted(point_indices_for_t)

        seeds = initialize_timepoint_seeds_with_prior(
            clusterer, time_index_to_linear_feature_indices, t, cluster_label2node, 1.2*len(timepoints), assigned_index_to_cluster
        )

        for idx, seed_info in tqdm(seeds.items(), desc="Looping through seed points", leave=False):
            # ancestor nodes (including the point node itself)
            # networkx.ancestors gives strict ancestors; include node itself:
            # ancestors = list(nx.ancestors(G, pt_idx))
            # ancestors.append(pt_idx)  # consider the candidate that is just the leaf itself
            # # Also consider ancestors sorted from nearest to farthest (optional)
            # # We compute depth by shortest path length from node to ancestor (if big tree, this is OK)
            # # To prefer smaller merges first, sort ancestors by increasing size (raw leaves).
            # ancestor_candidates = sorted(ancestors, key=lambda node: len(leaves_under.get(node, set())))
            anc = seed_info['start_node']
            start_indices = leaves_under[anc]

            is_updated = False
            if len(start_indices) == 1:
                print(f"Initializing a cluster from a leaf ({anc})")
                # Then we can form a new cluster
                best_candidate = None
                best_purity = -1.0
                best_kept_size = -1
            else:
                # Then we have a candidate hdbscan cluster, and need to calculate it's initial stats
                kept_indices, purity, time_map = compute_time_purity_for_indices(start_indices,
                                                                                 linear_ind_to_t_and_seg_id,
                                                                                 clusterer,
                                                                                 assigned_indices=set(assigned_index_to_cluster.keys()))
                best_candidate = (anc, start_indices, kept_indices, purity)
                best_purity = purity
                best_kept_size = len(kept_indices)
                # print(f"Initializing a cluster with hdbscan cluster ({seed_info['original_label']}, size={len(start_indices)})")
                
            
            for anc in tqdm(seed_info['ancestors'], desc="Checking merges", leave=False):
                raw_indices = leaves_under.get(anc, set())
                if len(raw_indices) == 0 or len(raw_indices) > 2*len(timepoints):
                    continue
                # Compute kept_indices and purity (prefer unassigned when tie via assigned_indices pass)
                kept_indices, purity, time_map = compute_time_purity_for_indices(raw_indices,
                                                                                 linear_ind_to_t_and_seg_id,
                                                                                 clusterer,
                                                                                 assigned_indices=set(assigned_index_to_cluster.keys()))
                kept_size = len(kept_indices)
                # We only consider candidates that will keep at least min_kept_size items
                if kept_size < min_kept_size:
                    continue

                # Candidate tie-breaking:
                # prefer higher purity; on equal purity prefer larger kept size
                if purity > best_purity + eps_purity_increase or (abs(purity - best_purity) <= eps_purity_increase and kept_size > best_kept_size):
                    best_candidate = (anc, raw_indices, kept_indices, purity)
                    best_purity = purity
                    best_kept_size = kept_size
                    is_updated = True

            # If we found a viable best candidate, check conflicts with already assigned indices
            if best_candidate is not None and best_purity > min_purity:
                anc_node, raw_indices, kept_indices, purity = best_candidate
                # Do not re-use indices already assigned to prior accepted clusters
                if any_assigned(kept_indices):
                    # skip candidate to keep clusters disjoint. Alternatively, we could drop assigned indices
                    # and recompute purity, but that adds complexity. For now we skip such candidates.
                    print(f"Found good cluster candidate, but it overlapped with an existing cluster; skipping")
                    continue

                # Accept this candidate cluster
                accepted_clusters.append({
                    'node_id': anc_node,
                    'raw_indices': set(raw_indices),
                    'kept_indices': set(kept_indices),
                    'purity': float(purity),
                })
                # mark kept indices as assigned
                for idx in kept_indices:
                    assigned_index_to_cluster[int(idx)] = cluster_counter
                # mark all raw but not-kept indices as outliers
                for idx in raw_indices:
                    if idx not in kept_indices:
                        outliers.add(idx)
                cluster_counter += 1

                if is_updated:
                    print(f"Accepted a modified cluster of size {len(kept_indices)}/{len(raw_indices)} with purity {purity} for object {idx} at t {t}")
                    num_clusters_changed += 1
                # else:
                #     print("Kept initialized cluster")
            else:
                print(f"No best candidate found (best_purity={best_purity}; size={best_kept_size})")
        # print("Stopping after t=0")
        # break

        print(f"In this iteration, {num_clusters_changed} clusters were modified or added (total accepted clusters: {cluster_counter})")
        if num_clusters_changed == 0:
            print("No clusters modified, stopping")

    return accepted_clusters, assigned_index_to_cluster, outliers