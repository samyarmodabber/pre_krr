import numpy as np
import scipy
import random

from sklearn.preprocessing import StandardScaler


def under_sample(X, y):
    """
        Balance the class distribution of the data X by under-sampling the over-represented class.

        Parameters
        ----------
        X : ndarray
            The data which is to be under-sampled.
        y : ndarray
            The target vector.

        Returns
        -------
        X : ndarray
            The balanced data.
        y : ndarray
            The corresponding target values of the balanced data.
        """
    print('Under-Sampling')
     # save label for all indices
    idx_pos = []
    idx_neg = []
    for i in range(len(y)):
        if y[i] == -1:
            idx_neg.append(i)
        else:
            idx_pos.append(i)

    # determine maximal number of samples per class for balanced subset
    num = min(len(idx_pos), len(idx_neg))

    r1 = random.sample(idx_pos, num)
    r2 = random.sample(idx_neg, num)
    r_samples = r1 + r2

    X = X[r_samples, :]
    y = y[r_samples]

    return X, y


def z_score_normalization(Xtrain, Xtest):
    """
    Z-score normalize the training and test data.

    Note
    ----
    Only the training data is included in fitting the normalizer to prevent train-test-contamination.

    Parameters
    ----------
    Xtrain : ndarray
        The training data which is to be z-score normalized and the normalizer is fitted on.
    Xtest : ndarray
        The test data which is to be z-score normalized based on the statistics from the training data.

    Returns
    -------
    X_train : ndarray
        Z-score normalized training data.
    X_test : ndarray
        Z-score normalized test data.
    """
    print('Z-Score Normalization')

    # z-score normalization

    # fit scaler only on train data to prevent train-test contamination
    scaler = StandardScaler()
    X_fit = scaler.fit(Xtrain)
    X_train = X_fit.transform(Xtrain)
    X_test = X_fit.transform(Xtest)

# =============================================================================
#     # scale data into [0,1] via the transformation x -> 0.5*(1+erf(x/sqrt(2)))
#     X_train = 0.5 * (1 + scipy.special.erf(X_train/np.sqrt(2)))
#     X_test = 0.5 * (1 + scipy.special.erf(X_test/np.sqrt(2)))
# =============================================================================

    return X_train, X_test


def data_preprocess(Xtrain, ytrain, Xtest, balance=True):
    """
    Preprocess the training and test data.

    Note
    ----
    Only the training data is included in fitting the normalizer to prevent train-test-contamination.

    Parameters
    ----------
    Xtrain : ndarray
        The training data which is to be balanced and z-score normalized and the normalizer is fitted on.
    ytrain : ndarray
        The corresponding target values of the training data which is to be balanced.
    Xtest : ndarray
        The test data which is to be z-score normalized based on the statistics from the training data.
    balance : bool, default=True
        Whether to balance the class distribution of the training data.

    Returns
    -------
    X_train : ndarray
        Balanced and z-score normalized training data.
    y_train : ndarray
        The corresponding target values of the balanced training data.
    X_test : ndarray
        Z-score normalized test data.
    """
    # balance the class distribution of the data by under-sampling the over-represenetd class
    if balance == True:
        Xtrain, y_train = under_sample(Xtrain, ytrain)

    # scale data with z-score-normalization
    X_train, X_test = z_score_normalization(Xtrain, Xtest)

    return X_train, y_train, X_test
