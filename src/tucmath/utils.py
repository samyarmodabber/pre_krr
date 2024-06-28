import scipy.linalg as la
from scipy.sparse.linalg import cg
import numpy as np


def diag_inv(M):
    """
    Get a 2d diagonal matrix or diagonal vector of a matrix as a vector 
    and return invers of matrix in 2dim.
    """

    if M.ndim == 1:
        d = 1./M
    elif M.ndim == 2:
        d = 1./np.diag(M)
    return np.diag(d)

def WB_Identity(A, U, V, k):
    '''
    Computes the Woodbury identity for find inverse of matrix A + UV^T where A is a diagonal matrix.

    (A+UV)^-1 = A^-1-A^-1U(I+VA^-1U)^-1VA^-1

    Parameters
    ----------
    A (numpy.ndarray): N by N diagonal matrix.
    U (numpy.ndarray): N by k matrix.
    V (numpy.ndarray): k by N matrix.
    k (int): rank of the matrix A + UV^T.
    Returns
    -------
    (A+UCV)^-1 (numpy.ndarray): N by N matrix. Invers of A+UCV.
    '''
    A_inv_diag = 1./np.diag(A)  # note! A_inv_diag is a vector!
    B_inv = np.linalg.inv(np.eye(k) + (V * A_inv_diag) @ U)
    return np.diag(A_inv_diag) - (A_inv_diag.reshape(-1, 1) * U @ B_inv @ V * A_inv_diag)


def CG(A, b, P_inv=None, matvec=None, tol=1e-5, x0=None, max_iter=None):
    """
    Conjugate Gradient Method for solving Ax = b
    :param A: matrix
    :param b: vector
    :param tol: tolerance (default: 1e-5)
    :param x0: initial guess (default: Zero vector)
    :param max_iter: Maximum number of iterations (default: Dimension of A)
    :param P_inv: The inverse of Preconditioner (default: None for CG)
    :param matvec: Function to perform matrix-vector multiplication (default: None)
    :return: x, residuals
    """

    N = len(b)
    residuals = []
    dot = np.dot

    if max_iter is None:
        max_iter = N

    if matvec is None:
        def matvec(M, u):
            return M @ u

    x = np.zeros(N) if x0 is None else x0
    r = b - matvec(A, x)
    p = r.copy()

    if P_inv is not None:
        z = P_inv @ r
        p = z.copy()
    else:
        z = r

    i = 0
    while i < max_iter and la.norm(r) > tol:
        residuals.append(np.linalg.norm(r))

        if la.norm(r) < tol:
            break

        v = matvec(A, p)

        if P_inv is None:
            alpha = dot(r, r) / dot(p, v)
        else:
            alpha = dot(r, z) / dot(p, v)

        x = x + alpha * p
        r_new = r - alpha * v

        if P_inv is None:
            beta = dot(r_new, r_new) / dot(r, r)
        else:
            z_new = P_inv @ r_new
            beta = dot(z_new, r_new) / dot(z, r)
            z = z_new

        p = z + beta * p if P_inv is not None else r_new + beta * p
        r = r_new
        i += 1

    return x, np.array(residuals)



def kernel(my_kernel, A=None, b=None, gamma=1.):
    '''
    This function takes matrix A and array(matrix) b as input and returns vector k(A,b)=[k(a,b) for row a in A]

    Parameters
    ----------
    A: ndarray: (N,d) matrix
    b: ndarray: (d,) array or (N,d) matrix

    Returns
    -------
    c: ndarray: (N,) array or (N,d) matrix c=A@b

    '''
    result = None
    if b.shape[0] == A.shape[1]:
        matrix = []
        for i in range(A.shape[0]):
            matrix.append(my_kernel(A[i, :].reshape(
                1, -1), b.reshape(1, -1), gamma)[0][0])
        result = np.array(matrix)
    elif b.shape[1] == A.shape[1]:
        result = my_kernel(A, b, gamma)
    return result
