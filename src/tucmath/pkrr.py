import numpy as np
from utils import CG
from sklearn.metrics.pairwise import rbf_kernel
from low_rank_methods import rpcholesky, greedy, nystrom, rff
from scipy.sparse.linalg import cg
from copy import copy


class PKRR:
    """
    Preconditioned Method for Kernel Ridge Regression

    Parameters
    ----------
    mu : float, default=0.1
        The regularization parameter.
    kernel : str, default="gaussian"
        The kernel function.
    gamma : float, default=1.0
         gamma parameter for the Gaussian kernel.
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
    >>> model1 = PKRR(mu=.5, rank=10, gamma=.5, prec="rpc")
    >>> model1.fit(X_train=X_train, y_train=y_train, max_iter=100)
    >>> print(model1.report)
    >>> y_predict_1 = model1.predict(X_test)
    >>> print(
    >>> f"Correct: {sum(y_test==y_predict_1 )} , Incorrect: {sum(y_test!=y_predict_1)}")
    """

    def __init__(self, mu=0.1, kernel="gaussian", gamma=1., prec=None, rank=200, tolerence=1e-3):

        self.mu = mu
        self.kernel = kernel
        self.gamma = gamma  # for kernel bandwidth
        self.rank = rank  # For low-rank approximation
        self.prec = prec  # precnditioner Name
        self.tolerence = tolerence

        self.K = None
        self.K_reg = None

        # For CG method
        self.solution = None
        self.residuals = None
        self.report = ""

        # For predict
        self.X_train = None
        self.k_X_train = None
        self.prediction = None

    ############################################################################

    def kernel_matrix(self):
        # Setup Kernel matrix
        if self.kernel == "gaussian":
            # K(x, y) = exp(-gamma ||x-y||^2)
            K = rbf_kernel(self.X_train, gamma=self.gamma)
        return K

    ############################################################################

    def matvec(self, x):
        return self.K_reg@x

    ##############################################################################

    def fit(self, X_train, y_train, maxiter=None):
        self.X_train = X_train
        N, d = X_train.shape
        I = np.eye(N)
        I_k = np.eye(self.rank)
        if maxiter is None:
            maxiter = N

        # Find Kernel and Regularize kernel: K+mu*I
        self.K = self.kernel_matrix()
        self.K_reg = self.K + self.mu*I

        #####################
        # PRECONDITIONING
        #####################
        if self.prec == "greedy":
            # GREEDY-BASED PIVOTED CHOLESKY APPROACH
            U, S, rows = greedy(self.K, self.rank)

        ########################
        elif self.prec == "rpc":
            # RANDOMIZED PIVOTED CHOLESKY APPROACH:
            U, S, rows = rpcholesky(self.K, self.rank)

        ###########################
        elif self.prec == "nystrom":
            # Nyström APPROACH
            U, D = nystrom(self.K, self.rank)

        ######################
        # preconditioning with random fourier features
        elif self.prec == "rff":
            # RANDOM FOURIER FEATURES APPROACH
            U, W, b = rff(self.X_train, rank=self.rank)
        
        elif self.prec is None:
            pass

        ######################
        # Not in methods for wrong typing
        else:
            raise ValueError(
                'Select precondition method from "rpc","greedy","rff", or "nystrom"')

        residuals = []

        A = copy(self.K_reg)
        b = copy(y_train)
        tol = copy(self.tolerence)

        if self.prec is None:
            # without preconditioning - Vanilla CG method
            def precon(x): return x
        else:
            # Preconditioning
            def precon(x):
                return x-U@np.linalg.solve(I_k+U.T@U, U.T@x)

        solution, residuals = CG(self.matvec, precon, b,tol=tol, max_iter=maxiter)
        self.solution = solution
        self.residuals = residuals

        method = "without precondition" if self.prec == None else self.prec
        self.report = f"Training is done in {len(residuals)} iteration with CG-method. Precondition: {method}"

    ##############################################################################
