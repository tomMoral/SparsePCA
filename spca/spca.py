import itertools
import numpy as np
from joblib import delayed, Parallel

from .utils import compute_L_support


def compute_spca_brute_force(B, k, n_jobs=1):
    """Compute the k-sparse PCA of B with a brute force algorithm.

    Note that this algorithm scales exponentially with the dimension of B so it
    should only be used for very small problems.

    Parameters
    ----------
    B : ndarray, shape (n_dim, n_dim)
        Matrix to extract the sub-matrix from
    k : int
        Sparsity level for the lower bound.
    n_jobs : int (default: 1)
        Number of parallel workers used to compute the estimation. Should be
        larger than 1.

    Return
    ------
    L_k: the k-sparse PCA of B with level k.
    """
    n_dim = B.shape[1]
    iterator = itertools.combinations(range(n_dim), k)
    if n_jobs > 1:
        L_S = Parallel(n_jobs=n_jobs, batch_size='auto')(
            delayed(compute_L_support)(B, support)
            for support in iterator
        )
    else:
        L_S = 0
        for support in iterator:
            L_S = max(L_S, compute_L_support(B, support))
    return np.max(L_S)
