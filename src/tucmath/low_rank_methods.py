from scipy.stats import cauchy, laplace
from warnings import warn
import numpy as np
from scipy import linalg as la
from scipy.linalg import cholesky, orth

from scipy.sparse import csc_matrix , csr_matrix 

def rpcholesky(K, rank, block=1):
    '''
    RPcholesky decomposition.
    Parameters
    ----------
    K : ndarray
        n by n SPD matrix.
    rank : int
        Desired rank.
    block : int
        Block size. default is 1 for non block version.
    Returns
    -------
    F : ndarray
        n by r Factor matrix. K=F@F.T
    S : list
        Index set.
    rows: list
        Rows on pivot.

    '''
    if block == 1:
        return cholesky_helper(K, rank, 'rp')
    else:
        return block_cholesky_helper(K, rank, block, 'rp')


def greedy(K, r, randomized_tiebreaking=False, block=1):
    '''
    Greedy decomposition.
    Parameters
    ----------
    K : ndarray
        n by n SPD matrix.
    r : int
        Desired rank.
    b : int
        Block size. Default is 1 for non block version.
    Returns
    -------
    F : ndarray
        n by r Factor matrix. K=F@F.T
    S : list
        Index set.
    rows: list
        Rows on pivot.

    '''
    if block == 1:
        return cholesky_helper(K, r, 'rgreedy' if randomized_tiebreaking else 'greedy')
    else:
        if randomized_tiebreaking:
            warn("Randomized tiebreaking not implemented for block greedy method")
        return block_cholesky_helper(K, r, block, 'greedy')


def uniform_old(A, k):
    N = A.shape[0]

    S = np.unique(np.random.choice(N, k, replace=False))

    if isinstance(A, np.ndarray):
        AS = A[:, S]
    else:
        AS = A(S)

    nu = np.finfo(float).eps * np.linalg.norm(AS, 'fro')  # Compute shift
    Y = AS + nu * csc_matrix((np.ones(k), (S, range(k))),shape=AS.shape).toarray()
    A_SS = Y[S, :]
    F = np.dot(Y, np.linalg.inv(cholesky(A_SS, lower=True)))

    return F, AS, S

def nystrom(K, rank):
    N = K.shape[0]
    G = np.random.randn(N, rank)
    Y = K@G  # matvec
    Q = np.linalg.qr(Y)[0]
    Ys = K@Q  # matvec
    C = Q.T @ Ys

    # compute LDL^T decomposition
    LL, D, per = la.ldl(C)
    D = D.clip(min=1e-2)
    L = LL@la.sqrtm(D)

    Ldec = np.zeros((N, rank))
    Ldec = np.linalg.solve(L.T, Ys.T)
    U = Ldec.T

    return U, D

def uniform(A, k):
    """
    UNIFORM Uniform sampling for Nystrom approximation
    Optional arguments:
    1. N: size of matrix A. Value is only used if A is an implicit matrix, in
       which case N must be specified.
    """

    N = A.shape[0]

    S = np.unique(np.random.choice(N, k, replace=False))

    if isinstance(A, np.ndarray):
        AS = A[:, S]
    else:
        AS = A(S)

    nu = np.finfo(float).eps * np.linalg.norm(AS, 'fro')  # Compute shift
    Y = AS + nu * csr_matrix((np.ones(k), (S, np.arange(k))),
                             shape=(AS.shape[0], AS.shape[1])).toarray()
    A_SS = Y[S, :]
    F = np.linalg.solve(cholesky(A_SS, lower=True), Y.T).T

    return F, AS, S, nu

def rff(X_train, gamma, metric = "rbf", rank=30):
    """Return random Fourier features based on data X, as well as random
    variables W and b.
    https://github.com/hichamjanati/srf/blob/master/RFF-I.ipynb
    """
    d = X_train.shape[1]
    if metric == "rbf":
        W = np.sqrt(2*gamma)*np.random.normal(size=(rank, d))
    elif metric == "laplace":
        W = cauchy.rvs(scale=gamma, size=(rank, d))

    u = 2*np.pi*np.random.rand(rank)
    Z= np.sqrt(2/rank) * np.cos(((X_train).dot(W.T) + u[np.newaxis, :]))

    return Z, W, u

#################################################################################
################################# Helper Function ###############################
#################################################################################


def cholesky_helper(K, r, alg):
    n = K.shape[0]
    d = np.copy(K.diagonal())

    # row ordering, is much faster for large scale problems
    G = np.zeros((r, n))
    rows = np.zeros((r, n))
    rng = np.random.default_rng()

    S = []

    for i in range(r):
        if alg == 'rp':
            s = rng.choice(range(n), p=d / sum(d))
        elif alg == 'rgreedy':
            s = rng.choice(np.where(d == np.max(d))[0])
        elif alg == "greedy":
            s = np.argmax(d)
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        S.append(s)
        rows[i, :] = K[s, :]
        G[i, :] = (rows[i, :] - G[:i, s].T @ G[:i, :]) / np.sqrt(d[s])
        d -= G[i, :]**2
        d = d.clip(min=0)

    return G.T, np.array(S), rows


def block_cholesky_helper(A, k, b, alg):
    d = np.copy(A.diagonal())
    n = A.shape[0]

    # row ordering
    G = np.zeros((k, n))
    rows = np.zeros((k, n))

    rng = np.random.default_rng()

    S = []

    cols = 0
    while cols < k:
        block_size = min(k-cols, b)

        if alg == 'rp':
            s = rng.choice(range(n), size=2*block_size,
                           p=d / sum(d), replace=False)
            s = np.unique(s)[:block_size]
            block_size = len(s)
        elif alg == 'greedy':
            s = np.argpartition(d, -block_size)[-block_size:]
        else:
            raise RuntimeError("Algorithm '{}' not recognized".format(alg))

        S.extend(s)
        rows[cols:cols+block_size, :] = A[s, :]
        G[cols:cols+block_size, :] = rows[cols:cols +
                                          block_size, :] - G[0:cols, s].T @ G[0:cols, :]
        C = G[cols:cols+block_size, s]
        L = np.linalg.cholesky(C+np.finfo(float).eps *
                               np.trace(C)*np.identity(block_size))
        G[cols:cols+block_size,
            :] = np.linalg.solve(L, G[cols:cols+block_size, :])
        d -= np.sum(G[cols:cols+block_size, :]**2, axis=0)
        d = d.clip(min=0)

        cols += block_size

    return G.T, np.array(S), rows
