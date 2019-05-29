import argparse
import numpy as np
import matplotlib.pyplot as plt

from spca.spca import compute_spca_brute_force
from spca.lower_bounds import compute_spca_lower_bound_greedy
from spca.lower_bounds import compute_spca_lower_bound_sampling
from spca.upper_bounds import compute_spca_upperbound_optimal

STYLES = {
    'brute': dict(c='C0', linewidth=3, label="k-SPCA"),
    'sampling': dict(c='C1', linestyle=(0, [3, 4]), linewidth=3,
                     label="sampling lower bound"),
    'upper': dict(c='C3', linestyle='--', linewidth=3, label="DSPCA"),
    'greedy': dict(c='C2', linestyle=(0, [3, 4]), linewidth=3,
                   label="greedy lower bound"),
}

METHODS = {
    'brute': compute_spca_brute_force,
    'greedy': compute_spca_lower_bound_greedy,
    'sampling': compute_spca_lower_bound_sampling,
    'upper': compute_spca_upperbound_optimal
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Experiments with Sparse PCA")
    parser.add_argument("--njobs", type=int, default=1,
                        help="Number of workers used for computations")
    args = parser.parse_args()

    run_methods = ['brute', 'upper', 'sampling', 'greedy']
    for method in run_methods:
        assert method in METHODS, f"Undefined method '{method}'"
        assert method in STYLES, f"You need to define a style for '{method}'"

    K, p = 25, 10
    random_state = 42
    np.random.seed(random_state)

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
            L_S = METHODS[method](B, k, n_jobs=args.njobs)
            if k > 1:
                # Make sure the results are monotonic
                L_S = max(L_S, results[method][-1])
            results[method].append(L_S)

    print(f"\rComputing k-SPCA bounds: done   ")

    for method in results:
        plt.plot(range(1, K+1), results[method], **STYLES[method])

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
