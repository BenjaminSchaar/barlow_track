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
from collections import Counter

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

def prepare_A_list_from_topk_arrays(probs_list, labels_list, N, K, normalize_rows=True, dtype=np.float64):
    """
    probs_list[r] : N x 3
    labels_list[r]: N x 3
    Returns: list of sparse CSR matrices (N x K) per run
    """
    A_list = []
    for r in range(len(probs_list)):
        probs = probs_list[r]
        labels = labels_list[r]
        rows = np.repeat(np.arange(N), 3)
        cols = labels.ravel()
        vals = probs.ravel().astype(dtype)
        if normalize_rows:
            row_sums = vals.reshape(N,3).sum(axis=1)
            vals = (vals.reshape(N,3) / row_sums[:,None]).ravel()
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N, K), dtype=dtype)
        A.sum_duplicates()
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

def aggregate_consensus(A_list, perms, K, doweight_runs=None):
    """
    Map each run's soft A_r into canonical label space using perms[r] and average.
    doweight_runs: optional array length R for weighting runs (e.g., reliability)
    Returns consensus_probs: N x K dense array (float)
    """
    R = len(A_list)
    N = A_list[0].shape[0]
    cons = np.zeros((N, K), dtype=float)
    if doweight_runs is None:
        doweight_runs = np.ones(R, dtype=float)
    for r, A in enumerate(A_list):
        perm = perms[r]  # length K with -1 allowed
        if A.nnz == 0:
            continue
        A_coo = A.tocoo()
        for i, j, v in zip(A_coo.row, A_coo.col, A_coo.data):
            mapped = perm[j]
            if mapped >= 0:
                cons[i, mapped] += doweight_runs[r] * v
            else:
                # mapped to null -> ignore or optionally put into an 'unknown' bucket
                pass
    # Optionally normalize per-row
    row_sums = cons.sum(axis=1, keepdims=True)
    nz = row_sums[:,0] > 0
    cons[nz] = cons[nz] / row_sums[nz]
    return cons

###########################
# 7) Full pipeline wrapper
###########################

def spectral_sync_from_topk(probs_list, labels_list, N, K,
                            do_col_weighting=True, weighting_strategy='invsqrt-count',
                            use_matrix_free=False, svd_params=None,
                            null_cost=None, normalize_input_rows=True, verbose=True):
    """
    runs_topk: list of run dicts mapping obj->[(label,prob), ...]
    N,K: number objects, labels
    Returns:
      perms: list (R) of mapping arrays (length K) with values -1..K-1
      consensus_probs: N x K array
      diagnostics dict
    """
    if svd_params is None:
        svd_params = dict(n_oversamples=20, n_iter=2)
    t0 = time.time()
    R = len(probs_list)

    # 1) build A_list
    if verbose:
        print("[info] Building per-run sparse matrices from top-k lists...")
    A_list = prepare_A_list_from_topk_arrays(probs_list, labels_list, N, K, normalize_rows=normalize_input_rows)

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
    consensus = aggregate_consensus(A_list, perms, K)

    # 6) diagnostics (overlap histogram & simple pairwise agreement sample)
    # overlap: how many runs label each object
    obj_counts = np.zeros(N, dtype=int)
    for A in A_list:
        obj_counts += (A.getnnz(axis=1) > 0).astype(int)
    obj_hist = np.bincount(obj_counts)
    obj_hist = obj_hist[obj_hist>0]
    diag = {
        'time': time.time() - t0,
        'obj_label_count_histogram': obj_hist.tolist(),
        'max_labels_per_object': int(obj_counts.max()),
        'min_labels_per_object': int(obj_counts.min()),
        'mean_labels_per_object': float(obj_counts.mean()),
        'V_shape': V.shape
    }
    if verbose:
        print("[info] Done. Diagnostics:", diag)

    return perms, consensus, diag

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

    # run spectral pipeline
    perms, consensus, diag = spectral_sync_from_topk(
        runs_topk, N, K, use_matrix_free=False, verbose=True
    )

    print("Example done. diag:", diag)
