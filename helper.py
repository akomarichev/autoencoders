import numpy as np
import cPickle
import codecs
from pylab import *
import matplotlib.pyplot as plt
import matplotlib


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


def save_image_w(W, numberOfPatches, patch_size, name):
    print name + "."
    k = 0
    fig = plt.figure()
    for i in range(numberOfPatches ** 2):
        for j in range(patch_size ** 2):
            ax = fig.add_subplot(patch_size * numberOfPatches, patch_size * numberOfPatches, k + 1)
            ax.imshow(W[:, j, i].reshape(patch_size, patch_size), cmap=cm.gray)
            plt.axis('off')
            k += 1
    fig.set_size_inches(25.6, 14.4)
    path = "images/" + name + ".png"
    fig.savefig(path, bbox_inches='tight', dpi=200)


def save_image_l(L, N, name):
    print name + "."
    k = 0
    fig = plt.figure()
    for i in range(N):
        ax = fig.add_subplot(np.sqrt(N), np.sqrt(N), k + 1)
        ax.imshow(L[:, :, i], cmap=cm.gray)
        plt.axis('off')
        k += 1
    fig.set_size_inches(25.6, 14.4)
    path = "images/" + name + ".png"
    fig.savefig(path, bbox_inches='tight', dpi=200)


def check_sparsity_of_gradients(grad, check_this):
    for item in grad:
        print "For ", item, " shape of gradients is: ", grad[item].shape
        print "Number of elements > 0: ", np.sum(grad[item] != 0)
        print "Number of elements equal to 0: ", np.sum(grad[item] == 0)
        print "Number of elememts at all: ", np.prod(grad[item].shape)
        print "Ratio: ", np.sum(grad[item] != 0)/(np.prod(grad[item].shape) + 0.0), '\n'
        if item != 'W4' and item == check_this:
            i, j, l = grad[item].shape
            for k in range(l):
                print "For patch ", k+1, ":"
                print "Number of elements > 0: ", np.sum(grad[item][:, :, k] != 0)
                print "Number of elements equal to 0: ", np.sum(grad[item][:, :, k] == 0)
                print "Number of elements at all: ", np.prod(grad[item][:, :, k].shape)
                print "Ratio: ", np.sum(grad[item][:, :, k] != 0)/(np.prod(grad[item][:, :, k].shape) + 0.0)
