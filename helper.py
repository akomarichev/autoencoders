import numpy as np
import cPickle


def calculate_k_sparsity_N(images, W, nOfPatches, patch_size, image_size, K, N):
    z = np.zeros((image_size, image_size, N))
    mask_z = np.zeros((image_size, image_size, N))
    buf = np.zeros((patch_size ** 2, N))
    mask_buf = np.zeros((patch_size ** 2, N))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            mask_buf.fill(0)
            buf = W[:, :, patchNumber].dot(images[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N))
            z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :] = buf.reshape(patch_size, patch_size, N)
            indexes = find_K_biggest_indices_general(buf, K)
            for i in range(N):
                mask_buf[indexes[:, i], i] = 1
            mask_z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :] = mask_buf.reshape(patch_size, patch_size, N)
            patchNumber += 1

    return z, mask_z


def k_sparsity_bp2_N(image, nOfPatches, patch_size, image_size, W, N):
    z = np.zeros((image_size, image_size, N))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :] = (W[:, :, patchNumber].T.dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N))).reshape(patch_size, patch_size, N)
            patchNumber += 1

    return z


def find_K_biggest_indices_general(a, K):
    return np.argsort(a, axis=0)[::-1, :][:K, :]


def pickle_data(file_name, data):
    f = open(file_name, 'wb')
    cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


def unpickle_data(file_name):
    f = open(file_name, 'rb')
    data = cPickle.load(f)
    f.close()
    return data
