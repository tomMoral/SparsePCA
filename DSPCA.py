import itertools
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt


from joblib import delayed, Parallel


N_JOBS = 6


def get_random_support(n_dim, sparsity, n_samples=10000):
    return np.array([np.random.choice(n_dim, sparsity, replace=False)
                     for _ in range(n_samples)])


def get_L_S(B, S):
    S = list(S)
    return np.linalg.norm(B[S][:, S], ord=2)


def upperbound_spca(B, k):
    n_dim = B.shape[0]
    X = cvx.Variable((n_dim, n_dim))
    pb = cvx.Problem(cvx.Maximize(cvx.trace(X * B)),
                     [cvx.constraints.PSD(X),
                      cvx.constraints.Zero(cvx.trace(X) - 1),
                      cvx.constraints.NonPos(cvx.sum(cvx.abs(X)) - k)
                      ])
    pb.solve(verbose=False, max_iters=10000)
    return pb.value


def sampling_lowerbound_spca(B, k, n_samples=10000):
    n_dim = B.shape[1]
    iterator = get_random_support(n_dim, k, n_samples)
    if N_JOBS > 1:
        batch_size = int(n_samples / (10 * N_JOBS))
        L_S = Parallel(n_jobs=N_JOBS, batch_size=batch_size)(
            delayed(get_L_S)(B, support)
            for support in iterator
        )
    else:
        L_S = 0
        for support in iterator:
            L_S = max(L_S, get_L_S(B, support))
    return np.max(L_S)


def greedy_lowerbound_spca(B, k):
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


def brute_force_spca(B, k):
    n_dim = B.shape[1]
    iterator = itertools.combinations(range(n_dim), k)
    if N_JOBS > 1:
        L_S = Parallel(n_jobs=N_JOBS, batch_size='auto')(
            delayed(get_L_S)(B, support)
            for support in iterator
        )
    else:
        L_S = 0
        for support in iterator:
            L_S = max(L_S, get_L_S(B, support))
    return np.max(L_S)


STYLES = {
    'brute': dict(c='C0', label="k-SPCA", linewidth=3),
    'lower': dict(c='C1', linestyle=(0, [3, 4]), label="sampling lower bound",
                    linewidth=3),
    'upper': dict(c='C3', linestyle='--', label="DSPCA", linewidth=3),
}

METHODS = {
    'brute': brute_force_spca,
    'lower': sampling_lowerbound_spca,
    'upper': upperbound_spca
}

if __name__ == "__main__":

    run_methods = ['brute', 'lower', 'upper']
    for method in run_methods:
        assert method in METHODS, f"Undefined method '{method}'"
        assert method in STYLES, f"You need to define a style for '{method}'"

    K, p = 25, 10
    D = np.random.randn(K, p)
    D /= np.linalg.norm(D, axis=1, ord=2, keepdims=True)
    B = D.dot(D.T)

    L = np.linalg.norm(B, ord=2)
    B /= L

    results = {k: [] for k in run_methods}

    for k in range(1, K+1):
        print(f"\rComputing k-SPCA bounds: {(k-1) / K:7.2%}", end='',
              flush=True)
        for method in run_methods:
            L_S = METHODS[method](B, k)
            if k > 1:
                # Make sure the results are monotonic
                L_S = max(L_S, results[method][-1])
            results[method].append(L_S)

    print(f"\rComputing k-SPCA bounds:   done")

    for k in results:
        plt.plot(results[k], **STYLES[k])

    xlim = (1, K)
    plt.hlines(1, *xlim, lineSTYLES='--', alpha=.7)
    plt.text((K + 1) / 4, 1, "L", va='center', ha='center',
             bbox=dict(facecolor='w', alpha=.5, edgecolor='w'))
    plt.xlim(xlim)
    plt.ylim(0, 1.05)
    plt.yticks([0, 1], ["0", "L"])
    plt.ylabel("$k$-SPCA value")
    plt.xlabel("sparsity")
    plt.legend()
    plt.savefig("SPCA.png")
    plt.show()
