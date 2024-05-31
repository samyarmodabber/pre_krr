import numpy as np
import pandas as pd
import random
import time
import scipy as sp
from utils import cg_iterations, WB_Identity, kernel
from sklearn.metrics.pairwise import rbf_kernel
from low_rank_methods import rpcholesky, greedy, nystrom, rff
from scipy.sparse.linalg import cg


class PKRR:
    """
    Preconditioned Method for Kernel Ridge Regression

    Parameters
    ----------
    mu : float, default=0.1
        The regularization parameter.
    kernel : str, default="gaussian"
        The kernel function.
    sigma : float, default=1.0
         Sigma parameter for the Gaussian kernel.
    rank : int, default=200
        The desired rank of the preconditioner.
    pre: str, default=rpc (Random Pivote Cholesky)
        Abbreviation of method name.
        Options: rp, greedy, nystrom

    Attributes
    ----------
    alpha_fast : ndarray
        The dual-variable for the SVM-Model.
    Xtrain : ndarray
        The training data used to fit the model.
    ytrain : ndarray
        The corresponding target vector.

    Examples
    --------
    >>> from pkrr import PKRR
    >>> model1 = PKRR(mu=.5, rank=10, sigma=0.1, gamma=.03, prec="rpc")
    >>> model1.fit(X_train=X_train, y_train=y_train, max_iter=100)
    >>> print(model1.report)
    >>> y_predict_1 = model1.predict(X_test)
    >>> print(
    >>> f"Correct: {sum(y_test==y_predict_1 )} , Incorrect: {sum(y_test!=y_predict_1)}")
    """

    def __init__(self, mu=0.1, kernel="gaussian", gamma=1., sigma=1., prec=None, rank=200, tolerence=1e-3):

        self.mu = mu
        self.kernel = kernel
        self.gamma = gamma  # for kernel bandwidth
        self.K = None
        self.K_reg = None
        self.sigma = sigma  # for rff

        # self.k = None
        self.rank = rank  # For low-rank approximation

        # self.I_k = np.eye(rank)
        # self.C = self.I_k
        # self.N = None
        # self.U = None

        # For precnditioner
        self.prec = prec  # Name
        self.Preconditioner = None  # M: Matrix

        # For CG method
        self.solution = None
        self.residuals = None
        self.tolerence = tolerence
        self.report = ""

        # For predict
        self.X_train = None
        self.k_X_train = None
        self.prediction = None

    ############################################################################

    def kernel_matrix(self):
        """
        Set up kernel matrix.

        Parameters
        ----------
        X_train : ndarray : The train data.

        Returns
        -------
        K : ndarray : kernel matrix.
        """
        # Setup Kernel matrix
        if self.kernel == "gaussian":
            # K(x, y) = exp(-gamma ||x-y||^2)
            K = rbf_kernel(self.X_train, gamma=self.gamma)
        return K

    ############################################################################

    def matvec(self, A, p):
        """
        Approximate matrix-vector product A*p

        Parameters
        ----------
        A : object
            The adjacency matrix object.
        p : ndarray : The vector, whose product A*p with the matrix A shall be approximated.

        Returns
        -------
        Ap : ndarray : The approximated matrix-vector product A*p.
        """
        return A@p

    ##############################################################################

    def fit(self, X_train, y_train, max_iter=None):
        """
        Perform the KRR for training the SVM on the train data.

        Parameters
        ----------
        X_train : ndarray :            The train data.
        y_train : ndarray :            The corresponding target vector.
        prec : str
            The preconditioner that shall be used for the IPM.
        max_iter : int
            The maximum number of interior point iterations.

        Returns
        -------
        alpha_fast : ndarray :            The learned classifier parameter.
        GMRESiter_fast : list
            Number of GMRES iterations within the interior points iterations.
        """
        self.X_train = X_train
        self.N = X_train.shape[0]

        # Find Kernel and Regularize kernel: K+mu*I
        self.K = self.kernel_matrix()
        self.K_reg = self.K+self.mu*np.eye(self.N)

        #####################
        # PRECONDITIONING
        #####################
        if self.prec == "greedy":
            # GREEDY-BASED PIVOTED CHOLESKY APPROACH
            F, S, rows = greedy(self.K, self.rank)
            self.Preconditioner = WB_Identity(self.K_reg, F, F.T, self.rank)

        ########################
        elif self.prec == "rpc":
            # RANDOMIZED PIVOTED CHOLESKY APPROACH:
            # Factor matrix
            F, S, rows = rpcholesky(self.K, self.rank)
            
            # Paper: Robust, randomized preconditioning for kernel ridge regression
            # Perform Economy Size SVD
            # U, D, Vt = sp.linalg.svd(F, full_matrices=False)
            # mu = self.mu
            # N, d = U.shape
            # I = np.eye(N)
            # i = np.eye(d)
            # inv = np.linalg.inv(D**2*i+self.mu*i)
            # self.Preconditioner = U@(inv-i/mu)@U.T+I/mu

            # Woodbury Identity
            self.Preconditioner = WB_Identity(self.K_reg, F, F.T, self.rank)

        ###########################
        elif self.prec == "nystrom":
            # Nyström APPROACH
            # setup Nyström decomposition
            F, D = nystrom(self.K, self.rank)
            self.Preconditioner = WB_Identity(self.K_reg, F, F.T, self.rank)

            # i = np.eye(D.shape[0])
            # I = np.eye(U.shape[0])
            # lamda_r = D[-1, -1]
            # self.Preconditioner = (self.mu+lamda_r)*U@(D+self.mu*i)@U.T+I-U@U.T #Paper

            # self.Preconditioner = WB_Identity(U, self.mu)

        ######################
        # preconditioning with random fourier features
        elif self.prec == "rff":
            # RANDOM FOURIER FEATURES APPROACH
            F, W, b = rff(self.K, self.rank, self.sigma)
            self.Preconditioner = WB_Identity(self.K_reg, F, F.T, self.rank)

        ######################
        # without preconditioning - vanilla CG method
        elif self.prec == None:
            self.Preconditioner = None
        ######################
        # Not in methods for wrong typing
        else:
            raise ValueError(
                'Select precondition method from "rpc","greedy","rff", or "nystrom"')

        # if self.prec != None:
        #     print(f"Condition number after apply {self.prec} (P@A):",
        #           np.linalg.cond(self.Preconditioner@K_reg))

        solution, residuals = cg_iterations(self.K_reg, y_train, M=self.Preconditioner,
                                            tol=self.tolerence)

        self.solution = solution
        self.residuals = residuals

        method = "without precondition" if self.prec == None else self.prec
        self.report= f"Training is done in {residuals.shape[0]} iteration with CG-method. Precondition: {method}"

    ##############################################################################

    def predict(self, x):
        """
        Predict class affiliations for the test data.

        Parameters
        ----------
        X_test : ndarray : The test data.

        Returns
        -------
        yhat_fast : ndarray : The predicted class affiliations for the test data.
        """
        M = kernel(rbf_kernel, self.X_train, x, gamma=self.gamma)
        predict0 = self.solution @ M

        predict = np.sign(predict0)
        return predict
