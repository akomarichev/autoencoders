import numpy as np
import scipy.io
import os.path

import helper


def whitening(X):
    print "Whitening! \n"
    print "X.shape: ", X.shape

    # X_sub = X[:, :, :100]
    # for i in range(10 ** 2):
    #     subplot(10, 10, i + 1), imshow(X_sub[:, :, i], cmap=cm.gray)
    # show()

    X = X.reshape(X.shape[1] ** 2, X.shape[2])

    X_mean = np.mean(X, axis=1)
    print "X_mean.shape: ", X_mean.shape
    X -= np.tile(X_mean, (X.shape[1], 1)).T

    sigma = X.dot(X.T)/X.shape[1]

    (U, S, V) = np.linalg.svd(sigma)

    X_rot = U.T.dot(X)

    # k = 0
    # for k in range(S.shape[0]):
    #     if S[0:k].sum() / S.sum() >= 0.99:
    #         break
    # print 'Optimal k to retain 99% variance is:', k

    # X_tilde = U[:, :k].T.dot(X)

    epsilon = 0.1

    X_pcawhite = np.diag(1.0 / np.sqrt(S + epsilon)).dot(X_rot)
    X_zcawhite = U.dot(X_pcawhite)

    X_zcawhite = X_zcawhite.reshape(np.sqrt(X.shape[0]), np.sqrt(X.shape[0]), X.shape[1])

    # X_sub = X_zcawhite[:, :, :100]
    # for i in range(10 ** 2):
    #     subplot(10, 10, i + 1), imshow(X_sub[:, :, i], cmap=cm.gray)
    # show()

    return X_zcawhite

    # print np.repeat(X_mean, X.shape[2]).shape


def load_data(path):
    data = scipy.io.loadmat(path)
    X_original = data['X']
    y = data['y'].flatten()

    row, col, rgb, m = X_original.shape
    X = np.zeros((m, row, col))

    # converting RGB into grayscale format
    for index in range(m):
        r, g, b = X_original[:, :, 0, index], X_original[:, :, 1, index], X_original[:, :, 2, index]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        X[index, :, :] = gray

    return X, y


def normalize_and_pickle(X, y, X_test, y_test):
    file_train = "data/pickles/train.pickle"
    file_train_labels = "data/pickles/labels_train.pickle"
    file_val = "data/pickles/val.pickle"
    file_val_labels = "data/pickles/labels_val.pickle"
    file_test = "data/pickles/test.pickle"
    file_test_labels = "data/pickles/labels_test.pickle"

    X_train = X[:72000, :, :]
    y_train = y[:72000]
    X_val = X[72000:, :, :]
    y_val = y[72000:]

    # subtract training mean from validation and test set
    mean = np.mean(X_train, axis=0)
    X_train -= mean
    X_val -= mean
    X_test -= mean

    X_train /= np.std(X_train, axis=0)
    X_val /= np.std(X_val, axis=0)
    X_test /= np.std(X_test, axis=0)

    print "Max (X_train): ", np.amax(X_train)
    print "Min (X_train): ", np.amin(X_train)
    print "Max (X_test): ", np.amax(X_test)
    print "Min (X_test): ", np.amin(X_test)
    print "Max (X_val): ", np.amax(X_val)
    print "Min (X_val): ", np.amin(X_val)

    helper.pickle_data(file_train, X_train.T)
    helper.pickle_data(file_val, X_val.T)
    helper.pickle_data(file_test, X_test.T)
    helper.pickle_data(file_train_labels, y_train)
    helper.pickle_data(file_val_labels, y_val)
    helper.pickle_data(file_test_labels, y_test)
