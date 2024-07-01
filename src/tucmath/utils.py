import scipy.linalg as la
from scipy.sparse.linalg import cg
import numpy as np

def CG(matvec, precon, b, tol, max_iter, x0=None, verbose=False):
    if x0 is not None and np.linalg.norm(x0) != 0:
        x = x0
        r = b - matvec(x)
    else:
        x = np.zeros_like(b)
        r = b

    bnorm = np.linalg.norm(b)
    rnorm = bnorm
    z = precon(r)
    p = z
    residuals=[]
    for iter in range(max_iter):
        if verbose:
            print(f"{iter}\t{rnorm / bnorm:e}")
        residuals.append(rnorm / bnorm)

        v = matvec(p)
        zr = np.dot(z.T, r)
        eta = zr / np.dot(v.T, p)
        x = x + eta * p
        r = r - eta * v
        z = precon(r)
        gamma = np.dot(z.T, r) / zr
        p = z + gamma * p
        rnorm = np.linalg.norm(r)

        if rnorm <= tol * bnorm:
            break

    return x, residuals

