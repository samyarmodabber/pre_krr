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



def cg_iterations(A, b, M, tol=1e-3):
    """
    Return the number of CG iterations for solving the system: K @ alpha = Y.

    Parameters
    ----------
    A : ndarray
        The kernel matrix.
    b : ndarray
        The vector on the right-hand side of the linear equation to be solved.

    Returns
    -------
    solution : float
        Solution of Ax=b.
    residuals : ndarray
        List of residuals at each iteration.
    """
    # Initialize counter to get the number of iterations needed in the CG algorithm
    residuals = []

    def callback(x): return residuals.append(
        np.linalg.norm(A @ x - b) / np.linalg.norm(b))

    alpha, info = cg(A, b, M=M, tol=tol, callback=callback)

    return alpha, np.array(residuals)


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
