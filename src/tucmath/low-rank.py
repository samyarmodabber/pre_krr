import numpy as np
from scipy.linalg import cholesky, orth
from scipy.sparse import csc_matrix


def pivotselect(d, m):
    N = len(d)
    return np.unique(np.random.choice(N, m, replace=True, p=d / np.sum(d)))


def choleskybase(A, d, pivotselect, k, B, tol):
    # Define Afun based on whether A is a float or a function
    if isinstance(A, np.ndarray):
        def Afun(S): return A[:, S]
        d = np.diag(A)
    else:
        Afun = A

    N = len(d)
    k = min(k, N)
    F = np.zeros((N, k))
    AS = np.zeros((N, k))
    S = np.zeros(k, dtype=int)
    i = 0
    scale = 4 * max(d)

    while i < k:
        s = pivotselect(d, min(B, k - i))
        S[i:i + len(s)] = s
        AS_new = Afun(s)
        G = AS_new - F[:, :i] @ F[s, :i].T
        H = G[s, :]
        R = np.linalg.cholesky(H + max(np.trace(H), scale)
                               * np.finfo(float).eps * np.eye(H.shape[0]))
        F[:, i:i + len(s)] = np.linalg.solve(R.T, G.T).T
        AS[:, i:i + len(s)] = AS_new
        d = np.maximum(d - np.linalg.norm(F[:, i:i + len(s)], axis=1) ** 2, 0)
        i += len(s)
        if np.sum(d) < tol:
            F = F[:, :i]
            S = S[:i]
            AS = AS[:, :i]
            break

    return F, AS, S


def rpcholesky(A, k, B, tol, d=None):
    N = A.shape[0]
    # Define the sampler function
    def sampler(dd, m): return np.unique(
        np.random.choice(N, m, replace=True, p=dd))

    # Call the Cholesky base function
    F, AS, S = choleskybase(A, d, sampler, k, B, tol)

    return F, AS, S


def greedy(A, k, B, tol, d=None):
    def greedyselect(d, m):
        return np.argsort(d)[-m:]

    F, AS, S = choleskybase(A, d, greedyselect, k, B, tol)
    return F, AS, S


def uniform(A, k):
    N = A.shape[0]

    S = np.unique(np.random.choice(N, k, replace=False))

    if isinstance(A, np.ndarray):
        AS = A[:, S]
    else:
        AS = A(S)

    nu = np.finfo(float).eps * np.linalg.norm(AS, 'fro')  # Compute shift
    Y = AS + nu * csc_matrix((np.ones(k), (S, range(k))),
                             shape=AS.shape).toarray()
    A_SS = Y[S, :]
    F = np.dot(Y, np.linalg.inv(cholesky(A_SS, lower=True)))

    return F, AS, S, nu


def nystrom(A, k):
    N = A.shape[0]
    Omega = orth(np.random.randn(N, k))  # Generate test matrix
    Y = np.dot(A, Omega)  # Compute sketch
    nu = np.sqrt(N) * np.finfo(float).eps * \
        np.linalg.norm(Y, 2)  # Compute shift
    Y = Y + nu * Omega
    B = np.dot(Omega.T, Y)
    C = cholesky(B, lower=True)
    F = np.dot(Y, np.linalg.inv(C))  # Ahat = FF'

    return F, nu
