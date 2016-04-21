import numpy as np

import helper


def k_sparse_original_model(theta, lambda_, images, patch_size, image_size, N, K):

    W1 = theta['W1']
    W2 = theta['W2']

    # Feedforward
    z2 = W1.dot(images.reshape(image_size ** 2, N))
    z2_mask = np.zeros(shape=(z2.shape))
    indexes = helper.find_K_biggest_indices_general(z2, K)
    for i in range(N):
        z2_mask[indexes[:, i], i] = 1
    z2 *= z2_mask

    z3 = W2.dot(z2)
    h = z3.reshape(image_size, image_size, N)

    cost = np.sum((h - images) ** 2) / (2 * N)

    delta3 = -(images - h).reshape(image_size ** 2, N)
    delta2 = W2.T.dot(delta3)
    delta2 *= z2_mask

    W2_d = delta3.dot((z2).reshape(image_size ** 2, N).T) / N
    W1_d = delta2.dot((images).reshape(image_size ** 2, N).T) / N

    grad = {'W1': W1_d, 'W2': W2_d}

    return cost, grad