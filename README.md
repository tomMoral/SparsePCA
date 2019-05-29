# Sparse PCA

Experimenting with Sparse PCA


## Definition

The sparse PCA is define for a given matrix $X$ in R^{n x p} and an integer k < p such that

maximize v^TX^TXv such that ||v||_2 <= 1 and ||v||_0 <= k.

## Upper bound with DSPCA

The paper by d'Aspremont, Bach and El Ghaoui (2008) [1][1] propose a method to compute an upper bound of this value based on a convex relaxation. The convexe relaxation is here directly solved using [`cvxpy`](https://www.cvxpy.org/).

## Sampling lower bound

By definition, the value of the k-SPCA can be lower bounded as the largest singular value of any sub mamatrix X_s composed by the columns of X from set s, as long as the cardinal of the set s is smaller than k. Thus, to compute an lower bound, it is possible to sample supports randomly and take the maximal singular value of all the generated supports.

## Greedy lower bound

The paper [1][1] also propose to compute a greedy lower bound for the sparse PCA. Their algorithm, named PathSPCA give another bound by computing greedily a support that have a large sigular value.


## Comparison

These three bounds can be compare for small problems with the brute force approach. Here are the result for a random matrix of size 25 x 25 and rank 10.

![image](SPCA_bounds.png)



[1] : https://www.di.ens.fr/~aspremon/PDF/OptSPCA.pdf
