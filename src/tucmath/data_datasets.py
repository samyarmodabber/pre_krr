import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler


##############################################
# SUSY Dataset


def susy(num=0):
    """
    Read the SUSY Dataset and prepare it for usage.

    Note
    ----
    If num!=0, this function returns a dataset with num number of samples per class.

    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included.

    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The corresponding training target vector incorporating the labels.
    y_test : ndarray
        The corresponding test target vector incorporating the labels.
    """
    # read dataset
    df = pd.read_csv("data/SUSY.csv", header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # convert dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()

    # reshape y
    y = np.reshape(y, (X.shape[0],))

    # convert labels: 0 -> -1 and save label for all indices
    idx_signal = []
    idx_background = []
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1.0
            idx_background.append(i)
        else:
            idx_signal.append(i)

    if num != 0:
        r1 = random.sample(idx_signal, num)
        r2 = random.sample(idx_background, num)
        r_samples = r1 + r2

        X = X[r_samples, :]
        y = y[r_samples]

    # split data in train and test split
    Xtrain, Xtest, y_train, y_test = train_test_split(
        X, y, train_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_fit = scaler.fit(Xtrain)
    X_train = X_fit.transform(Xtrain)
    X_test = X_fit.transform(Xtest)

    return X_train, X_test, y_train, y_test

##############################################
# HIGGS Dataset


def higgs(num=0):
    """
    Read the HIGGS Dataset and prepare it for usage.

    Note
    ----
    If num!=0, this function returns a dataset with num number of samples per class.

    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included.

    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The corresponding training target vector incorporating the labels.
    y_test : ndarray
        The corresponding test target vector incorporating the labels.
    """
    # read dataset
    df = pd.read_csv("data/HIGGS.csv", header=None)

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    # convert dataframe to numpy array
    X = X.to_numpy()
    y = y.to_numpy()

    # reshape y
    y = np.reshape(y, (X.shape[0],))

    # convert label: 0 -> -1 and save label for all indices
    idx_higgs = []
    idx_else = []
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = -1
            idx_else.append(i)
        else:
            idx_higgs.append(i)

    if num != 0:
        r1 = random.sample(idx_higgs, num)
        r2 = random.sample(idx_else, num)
        r_samples = r1 + r2

        X = X[r_samples, :]
        y = y[r_samples]

    # split data in train and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.5, random_state=42)

    return X_train, X_test, y_train, y_test

##############################################
# cod-rna Dataset


def cod_rna(num=0):
    """
    Read the cod-rna dataset and prepare it for usage.

    Note
    ----
    If num!=0, this function returns a dataset with num number of samples per class.
    The training and validation data are merged and used as the training data.

    Parameters
    ----------
    num : int, default=0
        The number of samples to be included per class.
        If num=0, all samples are included.

    Returns
    -------
    X_train : ndarray
        The training data matrix.
    X_test : ndarray
        The test data matrix.
    y_train : ndarray
        The corresponding training target vector incorporating the labels.
    y_test : ndarray
        The corresponding test target vector incorporating the labels.
    """
    # determine the number of samples to be included per class in the training and the test set
    num_cod = int(num/2)

    # read training and validation data set
    [X_t, y_t] = load_svmlight_file('data/cod-rna-train.txt')

    [X_v, y_v] = load_svmlight_file('data/cod-rna-val.txt')

    # concatenate training and validation set to training data set
    X_train = np.concatenate((X_t.todense(), X_v.todense()), axis=0)
    y_train = np.hstack((y_t, y_v))

    # save label for all indices
    idx_pos_train = []
    idx_neg_train = []
    for i in range(len(y_train)):
        if y_train[i] == 1:
            idx_pos_train.append(i)
        else:
            idx_neg_train.append(i)

    if num != 0:
        r1_train = random.sample(idx_pos_train, num_cod)
        r2_train = random.sample(idx_neg_train, num_cod)
        r_samples_train = r1_train + r2_train

        X_train = X_train[r_samples_train, :]
        y_train = y_train[r_samples_train]

    # read test dataset
    [X_test, y_test] = load_svmlight_file('data/cod-rna-test.txt')

    X_test = X_test.todense()

    # save label for all indices
    idx_pos_test = []
    idx_neg_test = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            idx_pos_test.append(i)
        else:
            idx_neg_test.append(i)

    if num != 0:
        r1_test = random.sample(idx_pos_test, num_cod)
        r2_test = random.sample(idx_neg_test, num_cod)
        r_samples_test = r1_test + r2_test

        X_test = X_test[r_samples_test, :]
        y_test = y_test[r_samples_test]

    return X_train, X_test, y_train, y_test
