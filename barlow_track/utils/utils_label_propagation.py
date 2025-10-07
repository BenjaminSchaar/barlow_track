from pynndescent import NNDescent
from tqdm.auto import tqdm
import torch
from torch_geometric.nn.models import LabelPropagation
from torch_sparse import spmm
import plotly.express as px
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_knn_graph(X, k=20):
    """
    X: np.ndarray (N, d) embeddings
    k: number of neighbors
    Returns PyTorch Geometric edge_index
    """
    index = NNDescent(X, n_neighbors=k, metric="euclidean")
    neighbors, _ = index.neighbor_graph

    rows, cols = [], []
    for i in range(len(neighbors)):
        for j in neighbors[i]:
            rows.append(i)
            cols.append(j)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


# def make_seed_labels(time_index_to_linear_feature_indices, slice_t, num_timepoints):
#     """
#     time: np.ndarray (N,)
#     slice_t: which time slice to use as seeds
#     Returns y (torch tensor, N,), with -1 = unlabeled
#     """
#     y = -torch.ones(num_timepoints, dtype=torch.long)
#     mask = time_index_to_linear_feature_indices[slice_t]
#     for i, m in enumerate(mask):
#         if m >= len(y):
#             break
#         y[m] = i + 1 #torch.arange(mask.sum())  # unique cluster IDs per object
#     return y

def make_seed_labels_no_dict(mask, num_timepoints):
    """
    time: np.ndarray (N,)
    slice_t: which time slice to use as seeds
    Returns y (torch tensor, N,), with -1 = unlabeled
    """
    y = -torch.ones(num_timepoints, dtype=torch.long)
    for i, m in enumerate(mask):
        if m >= len(y):
            break
        y[m] = i + 1 #torch.arange(mask.sum())  # unique cluster IDs per object
    return y


def run_label_propagation(edge_index, y, num_layers=50, alpha=0.9):
    """
    edge_index: graph edges
    y: seed labels (-1 for unlabeled)
    """
    mask = (y != -1)  # seeds
    y_filled = y.clone()
    y_filled[~mask] = 0  # dummy for unlabeled
    
    lp = LabelPropagation(num_layers=num_layers, alpha=alpha)
    out = lp(y_filled, edge_index, mask=mask)  # (N, C)
    pred = out.argmax(dim=-1)
    return pred


def clamped_label_propagation(edge_index, y, num_layers=50, DEBUG=True):
    """
    edge_index: graph edges
    y: seed labels (-1 for unlabeled)
    """
    # build adjacency (row-normalized) -- PyG has utils for this
    from torch_geometric.utils import add_self_loops, degree
    
    edge_index, _ = add_self_loops(edge_index)
    row, col = edge_index
    deg = degree(row, dtype=torch.float)
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    # deg_inv = deg.pow(-1)
    # deg_inv[deg_inv == float('inf')] = 0
    # norm = deg_inv[row]

    # seed mask
    mask = (y != -1)
    num_classes = int(y.max().item() + 1)
    Y = torch.zeros(y.size(0), num_classes, device=y.device)
    Y[mask, y[mask]] = 1.0

    H = Y.clone()

    for _ in range(num_layers):
        # propagate: H <- normalized adjacency * H
        H = spmm(edge_index, norm, H.size(0), H.size(0), H)

        # clamp seeds back to one-hot
        H[mask] = Y[mask]
    if DEBUG:
        print(f"Final H shape: {H.shape}")
        print(f"Final H max: {H.max().item()}")
        print(f"Final H sum per row: {H.sum(dim=1)[:10]}")  # First 10 rows

    return H


# def multi_seed_propagation(X, slices, time_index_to_linear_feature_indices, k=20):
#     edge_index = build_knn_graph(X, k=k)
#     labelings = []
#     for t in tqdm(slices, desc="Propagating labels from seed times"):
#         y = make_seed_labels(time_index_to_linear_feature_indices, slice_t=t, num_timepoints=X.shape[0])
#         pred = run_label_propagation(edge_index, y)
#         labelings.append(pred.numpy())
#     return labelings


def run_label_propagation(edge_index, y, num_layers=50, alpha=0.95, return_top_k=1, prob_thresh=None, tau=0.01, softmax=True, DEBUG=False):
    """
    edge_index: graph edges
    y: seed labels (-1 for unlabeled)
    """
    mask = (y != -1)  # seeds

    # Threshold purely chance labelings
    if prob_thresh is None:
        prob_thresh = 1.1*(1.0 / mask.sum().numpy())

    if DEBUG:
        print(f"Seeds found: {mask.sum().item()}")
        print(f"Unique labels: {torch.unique(y[mask]).tolist()}")
        print("Probability threshold: ", prob_thresh)
    
    # lp = LabelPropagation(num_layers=num_layers, alpha=alpha)
    # out = lp(y_filled, edge_index, mask=mask)  # (N, C)
    out = clamped_label_propagation(edge_index, y, num_layers=num_layers, DEBUG=DEBUG)

    if return_top_k == 1:
        probs = torch.softmax(out, dim=-1)
        max_probs, pred_labels = torch.max(probs, dim=-1)
        
        # Set low-confidence predictions to -1
        pred_labels[max_probs < prob_thresh] = -1
        pred = out.argmax(dim=-1)
        return pred, max_probs
    else:
        if softmax:
            probs = torch.softmax(out / tau, dim=-1)
        else:
            # Generates very confident labels
            row_sums = out.sum(dim=-1, keepdims=True)
            probs = torch.where(row_sums > 0, out / row_sums, torch.zeros_like(out))
        
        top_probs, top_labels = torch.topk(probs, return_top_k, dim=-1)
        # Threshold small labels
        top_probs[top_probs < prob_thresh] = 0
        top_labels[top_probs < prob_thresh] = -1
        
        return top_labels, top_probs  # (N, k), (N, k)


def multi_seed_propagation(X, slices, time_index_to_linear_feature_indices, k=20, **kwargs):
    edge_index = build_knn_graph(X, k=k)
    labelings = []
    probabilities = []
    for t in tqdm(slices, desc="Propagating labels from seed times", leave=False):
        y = make_seed_labels_no_dict(time_index_to_linear_feature_indices[t], num_timepoints=X.shape[0])
        pred, probs = run_label_propagation(edge_index, y, **kwargs)
        labelings.append(pred.numpy())
        probabilities.append(probs.numpy())
    return labelings, probabilities



def fuse_labels_per_time(aligned_labelings, time_index_to_linear_feature_indices, DEBUG=False):
    """
    aligned_labelings: list of np.arrays (N,) from different runs
    time_index_to_linear_feature_indices: dict indicating indices of each time point
    time_points: iterable of all unique time points
    Returns:
        final_labels: np.array (N,) final label assignment
        confidence: np.array (N,) number of runs agreeing on each label
    """
    N = len(aligned_labelings[0])
    final_labels = -np.ones(N, dtype=int)
    confidence = np.zeros(N, dtype=float)

    # process each time slice separately
    for t, idx in tqdm(time_index_to_linear_feature_indices.items(), desc="Aligning labels per time point", leave=False):
        # idx = np.where(times == t)[0]  # indices of this time point
        if len(idx) == 0:
            continue

        # Stack votes for this time
        votes = np.stack([y[idx] for y in aligned_labelings], axis=1)  # (M, R)
        num_objects, num_labelings = votes.shape
        if DEBUG:
            print(votes.shape)
        
        # Unique labels across all runs at this time slice
        unique_labels = np.unique(votes[votes != -1])
        num_labels = len(unique_labels)
        label_to_col = {l: i for i, l in enumerate(unique_labels)}
        
        # Flatten object indices and their votes
        obj_idx = np.repeat(np.arange(num_objects), num_labelings)
        lab_vals = votes.ravel()
        
        valid = lab_vals != -1
        obj_idx = obj_idx[valid]
        lab_vals = lab_vals[valid]
        
        lab_idx = np.vectorize(label_to_col.get)(lab_vals)
        
        cm = np.zeros((num_objects, num_labels), dtype=int)
        np.add.at(cm, (obj_idx, lab_idx), 1)


        # # Build confusion matrix: objects x labels
        # cm = np.zeros((num_objects, num_labels), dtype=int)
        # for i in range(num_objects):
        #     for j in range(num_labelings):
        #         v = votes[i, j]
        #         if v != -1:
        #             cm[i, label_to_col[v]] += 1

        # Hungarian matching: maximize total votes
        row_ind, col_ind = linear_sum_assignment(-cm)  # negative to maximize
        for i_object, i_label in zip(row_ind, col_ind):
            t_global = idx[i_object]
            final_labels[t_global] = unique_labels[i_label]
            confidence[t_global] = float(cm[i_object, i_label]) / float(num_labelings)
            
            if DEBUG:
                print(i_object, unique_labels[i_label], float(cm[i_object, i_label]) / float(num_labelings), confidence[t_global])
            
        if DEBUG:
            fig = px.imshow(cm)
            fig.show()
            break

    return final_labels, confidence



def align_pair(ref_labels, y_new):
    """
    Align y_new to ref_labels using Hungarian matching, adding new labels only if necessary.
    Both arrays must have the same length (N), -1 indicates unlabeled.
    
    Returns:
        y_new_aligned: np.array of same length as y_new
        mapping: dict mapping original y_new labels to global labels
    """
    if len(ref_labels) != len(y_new):
        raise ValueError(f"ref_labels and y_new must have same length: {len(ref_labels)} vs {len(y_new)}")
    
    # valid positions where both ref and new are labeled
    valid = (ref_labels != -1) & (y_new != -1)
    if not np.any(valid):
        # nothing to match, just assign new IDs for non -1
        y_new_aligned = y_new.copy()
        mapping = {}
        new_label_id = ref_labels.max() + 1 if np.any(ref_labels != -1) else 0
        for lb in np.unique(y_new):
            if lb != -1:
                mapping[lb] = new_label_id
                y_new_aligned[y_new == lb] = new_label_id
                new_label_id += 1
        return y_new_aligned, mapping
    # Flatten object indices and their votes
    # obj_idx = np.repeat(np.arange(num_objects), num_labelings)
    # lab_vals = votes.ravel()
    
    # valid = lab_vals != -1
    # obj_idx = obj_idx[valid]
    # lab_vals = lab_vals[valid]
    
    # lab_idx = np.vectorize(label_to_col.get)(lab_vals)
    
    # cm = np.zeros((num_objects, num_labels), dtype=int)
    # np.add.at(cm, (obj_idx, lab_idx), 1)

    # build confusion matrix for Hungarian matching
    labels_ref = np.unique(ref_labels[valid])
    labels_new = np.unique(y_new[valid])
    n_ref = len(labels_ref)
    n_new = len(labels_new)

    # mapping from label to matrix index
    ref_idx = {l: i for i, l in enumerate(labels_ref)}
    new_idx = {l: i for i, l in enumerate(labels_new)}

    cm = np.zeros((n_ref, n_new), dtype=int)
    for i, j in zip(np.where(valid)[0], np.where(valid)[0]):
        r = ref_idx[ref_labels[i]]
        c = new_idx[y_new[i]]
        cm[r, c] += 1

    # Hungarian matching (maximize total votes)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {labels_new[c]: labels_ref[r] for r, c in zip(row_ind, col_ind)}

    # assign new label IDs for unmatched columns
    unmatched_new = set(labels_new) - set(mapping.keys())
    new_label_id = ref_labels.max() + 1 if np.any(ref_labels != -1) else 0
    for lb in unmatched_new:
        mapping[lb] = new_label_id
        new_label_id += 1

    # apply mapping
    y_new_aligned = np.array([mapping.get(v, -1) if v != -1 else -1 for v in y_new])

    return y_new_aligned, mapping


def align_all(labelings, time_index_to_linear_feature_indices):
    """
    Align multiple labelings using a rolling reference.
    Assumes all labelings are equal-length arrays (N,), -1 = unlabeled.
    
    Returns:
        aligned: list of np.arrays, each aligned to rolling reference
    """
    if not labelings:
        return []

    # Start with the first labeling as reference
    ref_labels = labelings[0].copy()
    all_aligned_labelings = [ref_labels]

    # Rolling alignment
    for y in tqdm(labelings[1:], desc="Aligning all labelings"):
        y_aligned, mapping = align_pair(ref_labels, y)
        all_aligned_labelings.append(y_aligned)

        # Do fusion of the reference with this new aligned labeling and all prior ones, to be used for the next iteration
        ref_labels, ref_confidence = fuse_labels_per_time(all_aligned_labelings, time_index_to_linear_feature_indices)
        
    return all_aligned_labelings, ref_labels, ref_confidence
