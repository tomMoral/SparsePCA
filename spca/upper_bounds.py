
import cvxpy as cvx


def compute_spca_upperbound_optimal(B, k, n_jobs=1):
    """Compute an upper bound of the k-sparse PCA of B.

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
    L_k: upper bound on the k-sparse PCA of B with level k.
    """
    n_dim = B.shape[0]
    X = cvx.Variable((n_dim, n_dim))
    pb = cvx.Problem(cvx.Maximize(cvx.trace(X * B)),
                     [cvx.constraints.PSD(X),
                      cvx.constraints.Zero(cvx.trace(X) - 1),
                      cvx.constraints.NonPos(cvx.sum(cvx.abs(X)) - k)
                      ])
    pb.solve(verbose=False, max_iters=10000)
    return pb.value
