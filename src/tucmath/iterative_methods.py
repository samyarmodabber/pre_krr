import scipy.linalg as la
import numpy as np
import copy
from typing import Tuple


def CG(A, b, P_inv=None, matvec=None, tol=1e-5, x0=None, max_iter=None):
    """
    Conjugate Gradient Method for solving Ax = b
    :param A: matrix
    :param b: vector
    :param tol: tolerance (default: 1e-5)
    :param x0: initial guess (default: Zero vector)
    :param max_iter: Maximum number of iterations (default: Dimention of A)
    :param P_inv:  The inveresidualse of Preconditioner (default: None for CG)
    :return: x, residuals
    """

    N = len(b)
    residuals = []
    dot = np.dot

    if max_iter is None:
        max_iter = N
    # if matvec is None:
    #     def mat_vec(A, u): return A@u
    #     matvec = mat_vec

    def matvec(M, u):
        return M@u

    # def norm2(u):
    #     return np.linalg.norm(u)

    x = np.zeros(N) if x0 == None else x0
    r = b - matvec(A, x) if x.any() else b.copy()
    p = r
    z = np.zeros(N)

    if (P_inv is None):
        method = "cg"
    else:
        method = "pcg"
        z = matvec(P_inv, r)

    print("A shape", A.shape)
    print("b shape", b.shape)
    print("r shape", r.shape)

    # Main Loop
    i = 0
    while (i < max_iter or la.norm(r) > tol) and np.isfinite(r).all():
        print(i, r)
        residuals.append(np.linalg.norm(r))

        if la.norm(r) < tol:
            return x, np.array(residuals)

        v = matvec(A, p)

        if method == "cg":
            alpha = float(dot(r, r) / dot(p, v))
        elif method == "pcg":
            alpha = float(dot(r, z) / dot(p, v))

        # Update x and r
        x += alpha * p
        r_new = r - alpha * v

        # Update x and r

        if method == "cg":
            beta = float(dot(r_new, r_new)/dot(r, r))
        elif method == "pcg":
            z_new = matvec(P_inv, r_new)
            beta = float(dot(z_new, r_new)/dot(z, r))

        p = r_new + beta * p
        r = r_new
        i = i+1

    return x, np.array(residuals)


def PCG(A, b, Preconditioner, max_iter, tolerance) -> Tuple[np.matrix, np.matrix]:
    """Solve system of linear equations."""

    i = 0
    x_vec = copy.deepcopy(x_vec)
    r = b - A @ x_vec
    p = Preconditioner(A, r)
    delta_new = r.T * p
    residuals = []

    while i < max_iter or np.linalg.norm(r) > tolerance:
        q_vec = A @ p
        alpha = float(delta_new/(p.T*q_vec))
        # numpy has some problems with casting when using += notation...
        x_vec = x_vec + alpha*p
        r = r - alpha*q_vec
        s_P_inv = Preconditioner(A, r)

        delta_old = delta_new
        delta_new = r.T*s_P_inv
        beta = delta_new/delta_old
        p = s_P_inv + float(beta)*p

        residuals.append(np.linalg.norm(r))
        i += 1
    return x_vec, residuals
