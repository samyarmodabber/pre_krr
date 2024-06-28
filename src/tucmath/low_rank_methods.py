from warnings import warn
import numpy as np
from scipy import linalg as la


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


def uniform(A, rank):
    n = A.shape[0]
    sample = np.random.choice(range(n), rank, False)
    rows = A[sample, :]
    core = rows[:, sample]
    return core, rows, sample


def nystromT(K, rank):
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


def nystrom(K, rank):
    N = K.shape[0]
    Omega = np.random.randn(N, rank)
    Y = K@Omega  #matvec
    eps = np.finfo(float).eps
    shift = np.linalg.norm(Y, "fro")*eps
    Y_shift = Y+shift*Omega
    C = np.linalg.cholesky(Omega.T@Y_shift)
    B = Y_shift @ np.linalg.inv(C)

    U, D, VT = np.linalg.svd(B, full_matrices=False)
    Sigma = np.diag(D)
    Lamda = np.maximum(0, Sigma@Sigma-shift*np.eye(rank))

    return U, Lamda


def rff(X, D, sigma):
    """Return random Fourier features based on data X, as well as random
    variables W and b.
    https://github.com/NMADALI97/Nystrom_Method_vs_Random_Fourier_Features/blob/master/RFF.py
    """
    N, d = X.shape
    W = np.sqrt(2*sigma)*np.random.normal(size=(D, d))
    b = 2*np.pi*np.random.rand(D)
    Z = np.sqrt(2/D)*np.cos((X.dot(W.T) + b[np.newaxis, :]))
    return Z, W, b

def rffP(X, rank, gamma):
    """
    Random Fourier Features (RFF) approximation for RBF kernel.

    Parameters:
    X (np.ndarray): Input data matrix of shape (n_samples, n_features).
    rank (int): Number of random Fourier features.
    gamma (float): Parameter for the RBF kernel.

    Returns:
    np.ndarray: Approximated feature map of shape (n_samples, D).
    np.ndarray: W.
    np.ndarray: b.
    """
    N, d = X.shape

    # Generate random weights from a normal distribution
    W = np.random.normal(0, np.sqrt(2 * gamma), (d, rank))

    # Generate random bias from a uniform distribution
    b = np.random.uniform(0, 2 * np.pi, rank)

    # Compute the random Fourier features
    Z = np.sqrt(2 / rank) * np.cos(X.dot(W) + b)

    return Z, W, b

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
