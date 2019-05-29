import numpy as np


def compute_L_support(B, support):
    """Compute the largest eigenvalue of the sub-matrix of B indexed by support

    Parameters
    ----------
    B : ndarray, shape (n_dim, n_dim)
        Matrix to extract the sub-matrix from
    support : list
        Indexes of the element of the support.

    Return
    ------
    L_S: the l-2 norm of the submatrix B_{S, S} where S is the support.
    """
    support = list(support)
    return np.linalg.norm(B[support][:, support], ord=2)


def get_random_support(n_dim, sparsity, n_samples=10000):
    """Return a list of support sampled with the given sparsity level

    Parameters
    ----------
    n_dim : int
        Dimension of the original space
    sparsity : int
        Level of sparsity of the generated supports
    n_samples : int
        Number of supports drawn

    Return
    ------
    list_supports : ndarray, shape (n_samples, sparsity)
    """
    return np.array([np.random.choice(n_dim, sparsity, replace=False)
                     for _ in range(n_samples)])
