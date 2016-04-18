import numpy as np


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


def load_data_and_pickle(path, file_name_data, file_name_labels):
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

    X /= 255.0
    X = whitening(X.T)
    pickle_data(file_name_data, X)
    pickle_data(file_name_labels, y)
