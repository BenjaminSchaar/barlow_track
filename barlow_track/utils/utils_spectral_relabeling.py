"""
Spectral synchronization pipeline starting from top-3-per-object-per-run soft lists.

Input format required by `prepare_A_list_from_topk`:
  runs_topk : list of length R
    runs_topk[r] is a dict-like: mapping object_index -> list of (label, prob) for that object in run r.
    Example:
      runs_topk[0] = { 0: [(5,0.7),(3,0.2),(12,0.1)], 1: [(2,0.9)], ... }
    If a run didn't label object i, object i should not be present in the dict for that run.

Outputs:
  perms: list of length R; perms[r] is array length K with values in -1..K-1 mapping run-label -> canonical label (-1==null)
  consensus_probs: N x K dense (or sparse) array of per-object consensus probabilities in canonical label space
  optional diagnostics
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
import time
from tqdm.auto import tqdm


# try sklearn randomized_svd
try:
    from sklearn.utils.extmath import randomized_svd
    _HAS_RANDOMIZED = True
except Exception:
    _HAS_RANDOMIZED = False

from scipy.sparse.linalg import svds

###########################
# 1) Build A_list from top-3 data
###########################

def greedy_top1_timewise(labels_topk, probs_topk, time_index_to_linear_feature_indices):
    """
    Greedy top-1 matching within each time point.

    Args:
        labels_topk: list of NxK int tensors (or a single NxK tensor)
            - -1 indicates invalid label
        probs_topk: same shape as labels_topk, float probabilities
        time_index_to_linear_feature_indices: dict
            - time_index -> list of row indices corresponding to objects at that time

    Returns:
        labels_out: Nx1 int tensor, top-1 label per object
        probs_out: Nx1 float tensor, probability of chosen label
    """
    N, K = labels_topk.shape
    labels_out = np.full((N, 1), -1, dtype=np.int64)
    probs_out = np.zeros((N, 1), dtype=probs_topk.dtype)

    for t, rows in time_index_to_linear_feature_indices.items():
        # rows = torch.tensor(rows, dtype=torch.long)
        # Extract relevant rows
        row_labels = labels_topk[rows, 0]      # shape: len(rows) x K
        row_probs = probs_topk[rows, 0]        # same shape

        # Keep track of assigned labels at this time
        assigned_labels = set()
        sorted_idx = np.argsort(-row_probs)

        # Flatten the probabilities to pick the top-1 while respecting uniqueness
        # argsort descending
        for idx in sorted_idx:
            candidate_label = int(row_labels[idx])
            if candidate_label == -1:
                continue  # skip invalid labels
            if candidate_label not in assigned_labels:
                labels_out[rows[idx], 0] = candidate_label
                probs_out[rows[idx], 0] = row_probs[idx]
                assigned_labels.add(candidate_label)
            # If no valid label found, labels_out remains -1, probs_out=0

    return labels_out, probs_out



def prepare_A_list_from_labelings_probs(labelings, probabilities, N, time_index_to_linear_feature_indices, 
                                        normalize_rows=True, probability_threshold=0.0):
    """
    Converts a list of N x topk label index arrays + probability arrays into
    a list of N x K_total sparse matrices for spectral sync.
    
    labelings: list of N x topk candidate label arrays, 1-indexed, -1 means invalid
    probabilities: same shape as labelings
    normalize_rows: whether to normalize probabilities per object
    """
    topk = labelings[0].shape[1]
    A_list = []
    all_vals, all_rows, all_cols = [], [], []
    for L, P in zip(labelings, probabilities):
        L = np.array(L, dtype=int)
        P = np.array(P, dtype=float)
        P = np.nan_to_num(P, nan=0)
        P[P < probability_threshold] = 0.0

        # Do greedy matching to enforce temporal uniqueness
        if time_index_to_linear_feature_indices is not None:
            L, P = greedy_top1_timewise(L, P, time_index_to_linear_feature_indices)
            mask = L != -1
            rows = np.arange(N)[mask.ravel()]
            if any(L[mask]==0):
                raise ValueError("Assume 1-based indexing, but found label of 0")
            cols = (L[mask] - 1)  # convert 1-indexed -> 0-indexed
            vals = P[mask]

        else:
            # mask valid labels
            mask = L != -1
            rows = np.repeat(np.arange(N), topk)[mask.ravel()]
            cols = (L[mask] - 1)  # convert 1-indexed -> 0-indexed
            vals = P[mask]

        if normalize_rows:
            # compute row sums efficiently
            row_sums = np.bincount(rows, weights=vals, minlength=N)
            # avoid division by zero
            row_sums[row_sums == 0] = 1.0
            vals /= row_sums[rows]

        all_vals.append(vals)
        all_rows.append(rows)
        all_cols.append(cols)
    
    # Determine shape of matrices based on all max of all labels seen across labelings
    K = max([cols.max() if len(cols)>0 else 0 for cols in all_cols]) + 1
    for vals, rows, cols in zip(all_vals, all_rows, all_cols):
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, K))
        A_list.append(A)

    return A_list

###########################
# 2) Optional column weighting
###########################

def compute_column_weights(A_list, strategy='invsqrt-count', eps=1e-8):
    """
    Return vector length R*K of weights for concatenated S columns.
    strategy:
      - 'invsqrt-count': weight = 1/sqrt(count_col + eps)
      - 'inv-count': 1/(count_col+eps)
      - None: ones
    """
    R = len(A_list)
    N, K = A_list[0].shape
    weights = np.ones(R*K, dtype=float)
    base = 0
    for r, A in enumerate(A_list):
        counts = np.array(A.sum(axis=0)).ravel()  # sum per column (label)
        if strategy == 'invsqrt-count':
            w = 1.0 / np.sqrt(counts + eps)
        elif strategy == 'inv-count':
            w = 1.0 / (counts + eps)
        else:
            w = np.ones_like(counts, dtype=float)
        weights[base:base+K] = w
        base += K
    return weights

###########################
# 3) Build S or use matrix-free matvecs
###########################

def build_S(A_list, col_weights=None):
    """Return sparse CSR S = [A_0 | A_1 | ... | A_{R-1}] with optional column weighting (vector length R*K)"""
    S = sp.hstack(A_list, format='csr')
    if col_weights is not None:
        D = sp.diags(col_weights, offsets=0, format='csr')
        S = S.dot(D)
    return S

def matvec_S(A_list, X):
    """
    Compute S @ X when S = [A_0 ... A_{R-1}], X shape ((R*K), p)
    A_list: list of (N,K) sparse CSR
    X: np.array or dense matrix
    """
    R = len(A_list)
    K = A_list[0].shape[1]
    p = X.shape[1]
    out = None
    base = 0
    for r, A in enumerate(A_list):
        block = X[base:base+K, :]
        tmp = A.dot(block)  # N x p
        out = tmp if out is None else (out + tmp)
        base += K
    return out

def rmatvec_S(A_list, Y, left_multiply=False):
    """
    Compute S^T @ Y (when left_multiply=False; Y shape (N, p)) -> return (R*K, p)
    If left_multiply=True and Y is (N, P), return Q^T S (P x (R*K)) i.e. returns transpose ready for small SVD stage.
    """
    R = len(A_list)
    K = A_list[0].shape[1]
    blocks = []
    for r, A in enumerate(A_list):
        tmp = A.T.dot(Y)  # K x p
        blocks.append(tmp)
    out = np.vstack(blocks)
    if left_multiply:
        return out.T
    return out

###########################
# 4) Randomized SVD utilities (matrix and matrix-free)
###########################

def compute_topK_right_singular_vectors(S_or_A_list, K_dim, n_oversamples=20, n_iter=2, use_matrix_free=False):
    """
    If use_matrix_free=False: S_or_A_list is sparse matrix S (N x (R*K)) -> use randomized_svd or svds
    If use_matrix_free=True: S_or_A_list is A_list (and matvec functions used)
    Returns V ( (R*K) x K_dim ) right singular vectors (columns).
    """
    if not use_matrix_free:
        S = S_or_A_list
        if _HAS_RANDOMIZED:
            U, Sigma, Vt = randomized_svd(S, n_components=K_dim, n_oversamples=n_oversamples, n_iter=n_iter, random_state=0)
            V = Vt.T
            return V
        else:
            U, s, Vt = svds(S, k=K_dim, which='LM')
            V = Vt.T
            return V
    else:
        # matrix-free randomized SVD (simplified)
        A_list = S_or_A_list
        R = len(A_list)
        K = A_list[0].shape[1]
        RK = R * K
        rng = np.random.RandomState(0)
        P = K_dim + n_oversamples
        Omega = rng.normal(size=(RK, P))
        # Y = S * Omega
        Y = matvec_S(A_list, Omega)  # (N x P)
        for _ in range(n_iter):
            Z = rmatvec_S(A_list, Y)   # (RK x P)
            Y = matvec_S(A_list, Z)    # (N x P)
        # QR
        Q, _ = np.linalg.qr(Y, mode='reduced')  # N x P
        # B = Q^T S  => (P x RK)
        B = rmatvec_S(A_list, Q, left_multiply=True)  # P x RK
        # SVD of small B
        Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
        V = Vt.T[:, :K_dim]  # RK x K_dim
        return V

###########################
# 5) Rounding blocks to partial permutations (with NULL)
###########################

def round_block_to_partial_perm(V_block, null_cost=None):
    """
    V_block: K x K matrix. We perform a rectangular assignment (rows run-labels -> columns canonical labels + null).
    Returns mapping array of length K with values in -1..K-1
    """
    K = V_block.shape[0]
    # cost = -abs(similarity)
    C = -np.abs(V_block)
    # choose null cost adaptively if not provided
    if null_cost is None:
        best = np.max(np.abs(V_block), axis=1)
        median_best = np.median(best)
        # we set null cost slightly worse than typical good matches so only weak rows map to null
        null_cost = -0.5 * median_best
    null_col = np.full((K,1), null_cost)
    C_rect = np.hstack([C, null_col])  # K x (K+1)
    row_ind, col_ind = linear_sum_assignment(C_rect)
    mapping = np.full(K, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        if c < K:
            mapping[r] = int(c)
        else:
            mapping[r] = -1
    return mapping

###########################
# 6) Aggregate per-object consensus
###########################

def aggregate_consensus(A_list, perms, K, normalize_rows=True, doweight_runs=None):
    """
    Aggregate a list of N x K sparse matrices (one per run) into a consensus
    top-k labeling format via hungarian matching per time point.
    """
    R = len(A_list)
    N = A_list[0].shape[0]
    cons = np.zeros((N, K), dtype=float)
    if doweight_runs is None:
        doweight_runs = np.ones(R, dtype=float)
    for r, A in tqdm(enumerate(A_list), desc="Aggregating runs", total=len(A_list), leave=False):
        perm = perms[r]  # length K with -1 allowed
        if A.nnz == 0:
            continue
        A_coo = A.tocoo()
        for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
            mapped = perm[j]  # <-- Apply permutation here!
            if mapped >= 0:
                cons[i, mapped] += doweight_runs[r] * v
    return cons


def enforce_temporal_uniqueness_hungarian(consensus_probs, time_index_to_linear_feature_indices, 
                                          R, min_prob_threshold=None):
    """
    Postprocessing to enforce one-object-per-time-point using Hungarian matching.
    
    Args:
        consensus_probs: N x K array of consensus probabilities
        time_index_to_linear_feature_indices: dict mapping time_index -> list of object indices at that time
        return_format: 'topk' returns top-k labels/probs per object, 'full' returns full N x K matrix
        topk: number of top labels to return per object (only used if return_format='topk')
        min_prob_threshold: minimum probability to consider a label valid
        
    Returns:
        If return_format='topk':
            labels_out: N x topk array (1-indexed, -1 for invalid)
            probs_out: N x topk array of probabilities
        If return_format='full':
            consensus_probs_unique: N x K array with temporal uniqueness enforced
    """
    N, K = consensus_probs.shape
    if min_prob_threshold is None:
        min_prob_threshold = 0.02  # e.g. at least 2% of the runs agreeing
    
    # Create a copy to modify
    consensus_unique = consensus_probs.copy()
    probs_out = np.zeros((N, 1), dtype=float)
    labels_out = np.full((N, 1), -1, dtype=int)

    # Process each time point independently
    for time_idx, object_indices in tqdm(time_index_to_linear_feature_indices.items(), desc="Enforcing temporal uniqueness"):
        if len(object_indices) <= 1:
            continue  # No conflict possible with 0 or 1 objects
            
        # Extract probabilities for objects at this time point
        time_probs = consensus_probs[object_indices, :]  # shape: (n_objects, K)
        n_objects = len(object_indices)
        
        # Create cost matrix for Hungarian algorithm
        # Cost = -probability (since Hungarian finds minimum cost)
        cost_matrix = -time_probs
        
        # Handle case where we have more objects than labels or vice versa
        if n_objects > K:
            # More objects than labels - some objects will get no assignment
            # Pad cost matrix with high cost "null" assignments
            null_cost = -min_prob_threshold  # Small positive cost for null assignment
            cost_matrix = np.hstack([cost_matrix, 
                                   np.full((n_objects, n_objects - K), null_cost)])
        elif n_objects < K:
            # More labels than objects - some labels won't be assigned
            # Pad with dummy objects that have high cost for all labels
            dummy_cost = -min_prob_threshold
            dummy_rows = np.full((K - n_objects, K), dummy_cost)
            cost_matrix = np.vstack([cost_matrix, dummy_rows])
            
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Prepare outout as top-1 format, i.e. use the hungarian label but keep the probability from original
        for obj_idx, label_idx in zip(row_indices, col_indices):
            if obj_idx < n_objects and label_idx < K:
                probs_out[object_indices[obj_idx], 0] = time_probs[obj_idx, label_idx] 
                labels_out[object_indices[obj_idx], 0] = label_idx + 1
            
        # Clear all assignments for this time point first
        # consensus_unique[object_indices, :] = 0.0
        
    #     # Apply the Hungarian assignment
    #     for obj_idx, label_idx in zip(row_indices, col_indices):
    #         if obj_idx < n_objects and label_idx < K:
    #             # Valid assignment to a real object and real label
    #             original_prob = time_probs[obj_idx, label_idx]
    #             if original_prob > min_prob_threshold:
    #                 consensus_unique[object_indices[obj_idx], label_idx] = original_prob
        
    # Normalize probabilities by number of runs
    probs_out /= R

    # And remove extremely low probabilities
    invalid_mask = probs_out < min_prob_threshold
    probs_out[invalid_mask] = 0.0
    labels_out[invalid_mask] = -1
    
    # if return_format == 'full':
    #     return consensus_unique
    
    # elif return_format == 'topk':
    #     # Convert to top-k format
    #     labels_out = np.full((N, topk), -1, dtype=int)
    #     probs_out = np.zeros((N, topk), dtype=float)
        
    #     for i in range(N):
    #         row = consensus_unique[i, :]
    #         if np.all(row == 0):
    #             continue  # All probabilities are zero - leave as -1
                
    #         # Get top-k indices (in descending order)
    #         top_indices = np.argsort(row)[-topk:][::-1]
    #         valid_mask = row[top_indices] > min_prob_threshold
            
    #         valid_indices = top_indices[valid_mask]
    #         labels_out[i, :len(valid_indices)] = valid_indices + 1  # Convert to 1-indexed
    #         probs_out[i, :len(valid_indices)] = row[valid_indices]
            
    return labels_out, probs_out
    
    # else:
    #     raise ValueError(f"Unknown return_format: {return_format}")



###########################
# 7) Full pipeline wrapper
###########################

def spectral_sync_from_topk(labels_list, probs_list, K,
                            time_index_to_linear_feature_indices,
                            do_col_weighting=True, weighting_strategy='invsqrt-count',
                            use_matrix_free=False, svd_params=None,
                            null_cost=None, normalize_input_rows=True, input_probability_threshold=0.0, 
                            verbose=True, DEBUG=False):
    """
    runs_topk: list of run dicts mapping obj->[(label,prob), ...]
    K: number of labels; safe to overestimate
    Returns:
      perms: list (R) of mapping arrays (length K) with values -1..K-1
      consensus_probs: N x K array
      diagnostics dict
    """
    if svd_params is None:
        svd_params = dict(n_oversamples=20, n_iter=2)
    t0 = time.time()
    R = len(probs_list)
    N, topk = labels_list[0].shape
    if time_index_to_linear_feature_indices is not None:
        topk = 1

    # 1) build A_list
    if verbose:
        print("[info] Building per-run sparse matrices from top-k lists...")
        print(f"Found {len(labels_list)} labelings of shape {labels_list[0].shape}")
    A_list = prepare_A_list_from_labelings_probs(labels_list, probs_list, N, time_index_to_linear_feature_indices,
                                                 normalize_rows=normalize_input_rows,
                                                 probability_threshold=input_probability_threshold)
    if DEBUG:
        print(f"Refactored into {len(A_list)} sparse matrices of shape {A_list[0].shape}")
        for i, A in enumerate(A_list):
            # row_sums = np.array(A.sum(axis=1)).flatten()
            # print(A.shape, row_sums.shape)
            # zero_rows = np.where(row_sums == 0)[0]
            # print("Completely unlabeled objects:", zero_rows)
            A_dense = A.toarray()
            print(A_dense.min(), A_dense.max(), np.any(np.isnan(A_dense)), np.any(np.isinf(A_dense)))
            import plotly.express as px
            fig = px.imshow(A_dense[:1000, :].T, aspect=1)
            fig.show()
            print(np.where(np.isnan(A_dense)))
            if i>0:
                return

    # 2) column weighting
    col_weights = None
    if do_col_weighting:
        if verbose: print("[info] computing column weights...")
        col_weights = compute_column_weights(A_list, strategy=weighting_strategy)

    # 3) compute top-K right singular vectors (matrix or matrix-free)
    if verbose: print("[info] computing top-K right singular vectors (may be the heaviest step)...")
    if not use_matrix_free:
        S = build_S(A_list, col_weights)
        V = compute_topK_right_singular_vectors(S, K_dim=K,
                                                n_oversamples=svd_params['n_oversamples'],
                                                n_iter=svd_params['n_iter'],
                                                use_matrix_free=False)
    else:
        # apply weighting by scaling A_list columns in place (cheap)
        if col_weights is not None:
            # scale each A_list block's columns by corresponding weights
            base = 0
            for r in range(R):
                w = col_weights[base:base+K]
                # multiply each column j by w[j]  -> done using sparse diag
                A_list[r] = A_list[r].dot(sp.diags(w, format='csr'))
                base += K
        # matrix-free SVD
        V = compute_topK_right_singular_vectors(A_list, K_dim=K,
                                                n_oversamples=svd_params['n_oversamples'],
                                                n_iter=svd_params['n_iter'],
                                                use_matrix_free=True)

    if verbose:
        print(f"[info] SVD done: V shape {V.shape}. Time elapsed: {time.time()-t0:.1f}s")

    # 4) split into blocks and round
    perms = []
    for r in range(R):
        start = r*K
        block = V[start:start+K, :]  # K x K
        perm = round_block_to_partial_perm(block, null_cost=null_cost)
        perms.append(perm)

    # 5) aggregate consensus probabilities
    cons = aggregate_consensus(A_list, perms, K, normalize_rows=True)
    # labels_out = labels_out + 1 
    # labels_out[probs_out.sum(axis=1) == 0] = -1

    # 6) enforce temporal uniqueness
    labels_out, probs_out = enforce_temporal_uniqueness_hungarian(
        cons, time_index_to_linear_feature_indices, R=R
    )

    # Last - diagnostics (overlap histogram & simple pairwise agreement sample)
    # overlap: how many runs label each object
    obj_counts = np.zeros(N, dtype=int)
    for A in A_list:
        obj_counts += (A.getnnz(axis=1) > 0).astype(int)
    obj_hist = np.bincount(obj_counts)
    obj_hist = obj_hist[obj_hist>0]
    diag = {
        'time': time.time() - t0,
        # 'obj_label_count_histogram': obj_hist.tolist(),
        'max_labels_per_object': int(obj_counts.max()),
        'min_labels_per_object': int(obj_counts.min()),
        'mean_labels_per_object': float(obj_counts.mean()),
        'V_shape': V.shape
    }
    if verbose:
        print("[info] Done. Diagnostics:", diag)

    return perms, labels_out, probs_out, diag

###########################
# Example quick test (small) with incomplete runs
###########################
if __name__ == "__main__":
    # small synthetic example
    N = 2000
    K = 30
    R = 60
    rng = np.random.RandomState(1)
    
    # synth ground truth canonical assignment: a soft distribution per object
    gt = rng.dirichlet(alpha=np.ones(K)*0.5, size=N)
    
    # simulate runs: permute labels and reveal top-3 with noise, incomplete coverage
    runs_topk = []
    for r in range(R):
        perm = rng.permutation(K)
        rundict = {}
        # choose a random subset of objects for this run (70–95% coverage)
        coverage_fraction = rng.uniform(0.7, 0.95)
        num_objects = int(N * coverage_fraction)
        chosen_objects = rng.choice(N, size=num_objects, replace=False)
        for i in chosen_objects:
            # sample a label from gt for the object (as noisy observation)
            sampled_label = np.argmax(np.random.multinomial(1, gt[i]))
            # local label in this run
            local_label = int(np.where(perm == sampled_label)[0][0])
            # construct top 3: include true with high prob + two distractors
            distractors = rng.choice(K, size=2, replace=False)
            labs = [local_label, distractors[0], distractors[1]]
            probs = np.array([0.7, 0.2, 0.1])
            rundict[i] = list(zip(labs, probs))
        runs_topk.append(rundict)

    # Convert runs_topk to expected format
    labels_list = []
    probs_list = []
    for run_dict in runs_topk:
        labels = np.full((N, 3), -1)  # assuming top-3
        probs = np.zeros((N, 3))
        for obj_id, label_prob_list in run_dict.items():
            for k, (label, prob) in enumerate(label_prob_list):
                labels[obj_id, k] = label + 1  # convert to 1-indexed
                probs[obj_id, k] = prob
        labels_list.append(labels)
        probs_list.append(probs)

    # Then call with correct arguments
    perms, labels_out, probs_out, diag = spectral_sync_from_topk(
        labels_list, probs_list, K, None, verbose=True
    )

    print("Example done. diag:", diag)
