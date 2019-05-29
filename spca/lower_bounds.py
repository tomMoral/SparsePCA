import numpy as np
from joblib import delayed, Parallel


from .utils import compute_L_support
from .utils import get_random_support


def compute_spca_lower_bound_sampling(B, k, n_samples=10000, n_jobs=1):
    """Compute a lower bound of the k-sparse PCA of B.

    Parameters
    ----------
    B : ndarray, shape (n_dim, n_dim)
        Matrix to extract the sub-matrix from
    k : int
        Sparsity level for the lower bound.
    n_samples : int (default: 10000)
        Number of samples drawn to estimate the lower bound. The higher the
        better for the estimate quality but it also increase the runtime.
    n_jobs : int (default: 1)
        Number of parallel workers used to compute the estimation. Should be
        larger than 1.

    Return
    ------
    L_k: a lower bound on the k-sparse PCA of B with level k.
    """
    n_dim = B.shape[1]
    iterator = get_random_support(n_dim, k, n_samples)
    if n_jobs > 1:
        batch_size = int(n_samples / (10 * n_jobs))
        L_S = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(compute_L_support)(B, support)
            for support in iterator
        )
    else:
        L_S = 0
        for support in iterator:
            L_S = max(L_S, compute_L_support(B, support))
    return np.max(L_S)


def compute_spca_lower_bound_greedy(B, k, n_jobs=1):
    """Compute a lower bound of the k-sparse PCA of B.

    Parameters
    ----------
    B : ndarray, shape (n_dim, n_dim)
        Matrix to extract the sub-matrix from
    k : int
        Sparsity level for the lower bound.
    n_jobs : int (default: 1)
        Number of jobs used for computations. Here, this parameter has no
        effect.

    Return
    ------
    L_k: a lower bound on the k-sparse PCA of B with level k.
    """
    n_dim = B.shape[0]

    set_N = set(range(n_dim))
    support = []

    L_S = 0
    for s in range(k):
        remaining_idx = list(set_N - set(support))
        for i in remaining_idx:
            new_L_S = compute_L_support(B, support + [i])
            if new_L_S >= L_S:
                L_S = new_L_S
                new_support = support + [i]
        support = new_support

    return L_S


def path_spca(B, k):
    """Compute leading k-sparse principal component for matrix B"""
    M, N = B.shape

    set_N = set(range(N))

    # Loop through variables

    # For the first support, select the column with the largest norm
    Bs = (B * B).sum(axis=0)
    idx_k = Bs.argmax()
    support_k = [idx_k]
    Stemp = np.array([[Bs[idx_k]]])
    for i in range(1, k):
        # Compute x_k as the leading eigenvector of sum_{j in support} a_ja_j^T
        _, v = np.linalg.eigh(Stemp)
        x_k = B[:, support_k].dot(v[:, -1])
        x_k = x_k / np.linalg.norm(x_k, ord=2)

        # Compute the complement of the support and select max_i x_k^Ta_i
        comp_support_k = list(set_N - set(support_k))
        vals = x_k.T.dot(B[:, comp_support_k])
        idx_k = comp_support_k[vals.argmax()]

        # Update the Stemp matrix
        Stemp = np.column_stack((Stemp, B[:, support_k].T.dot(B[:, idx_k])))
        vbuf = np.append(B[:, idx_k].T.dot(B[:, support_k]),
                         np.array([(B[:, idx_k] * B[:, idx_k]).sum()]))
        Stemp = np.row_stack((Stemp, vbuf))

        # Update the support of the k-SPCA
        support_k.append(idx_k)

    lev, v = np.linalg.eig(Stemp)
    assert np.isclose(np.real(lev).max(),
                      np.linalg.norm(B[support_k][:, support_k], ord=2))
    return np.real(lev).max(), support_k
