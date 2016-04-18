import numpy as np


def initialize_k_deep_sparse_autoencoder(patch_size, image_size):
    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)
    numberOfPatches_l3 = image_size // (4 * patch_size)

    r = np.sqrt(6) / np.sqrt(patch_size ** 2 + patch_size ** 2 + 1)
    W1 = np.zeros((patch_size ** 2, patch_size ** 2, numberOfPatches_l1 ** 2))
    for i in range(numberOfPatches_l1 ** 2):
        W1[:, :, i] = np.random.random((patch_size ** 2, patch_size ** 2)) * 2 * r - r

    updated_patch_size = 2 * patch_size
    r = np.sqrt(6) / np.sqrt(updated_patch_size ** 2 + updated_patch_size ** 2 + 1)
    W2 = np.zeros((updated_patch_size ** 2, updated_patch_size ** 2, numberOfPatches_l2 ** 2))
    for i in range(numberOfPatches_l2 ** 2):
        W2[:, :, i] = np.random.random((updated_patch_size ** 2, updated_patch_size ** 2)) * 2 * r - r

    updated_patch_size = 4 * patch_size
    r = np.sqrt(6) / np.sqrt(updated_patch_size ** 2 + updated_patch_size ** 2 + 1)
    W3 = np.zeros((updated_patch_size ** 2, updated_patch_size ** 2, numberOfPatches_l3 ** 2))
    for i in range(numberOfPatches_l3 ** 2):
        W3[:, :, i] = np.random.random((updated_patch_size ** 2, updated_patch_size ** 2)) * 2 * r - r

    W4 = np.zeros((image_size ** 2, image_size ** 2))
    r = np.sqrt(6) / np.sqrt(image_size ** 2 + image_size ** 2 + 1)
    W4[:, :] = np.random.random((image_size ** 2, image_size ** 2)) * 2 * r - r

    # B1 = np.zeros((image_size, image_size))
    # B2 = np.zeros((image_size, image_size))
    # B3 = np.zeros((image_size, image_size))

    theta = {'W1': W1, 'W2': W2, 'W3': W3, 'W4': W4}

    return theta
