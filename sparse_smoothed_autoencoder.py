import numpy as np
import os.path
import gradient_check
import random
import scipy.io
import scipy.optimize
import scipy.sparse
from pylab import *
import matplotlib.pyplot as plt
import matplotlib
import PIL
import scipy.signal
import cPickle
import timeit

N = 10000
pool_size = 3
epsiln = 3e-3
image_size = 32
EPSILON = 0.0001

# weight initialization


def initialize(patch_size, image_size):
    numberOfPatches = image_size // patch_size

    r = np.sqrt(6) / np.sqrt(patch_size ** 2 + patch_size ** 2 + 1)
    W1 = np.zeros((patch_size ** 2, patch_size ** 2, numberOfPatches ** 2))
    for i in range(numberOfPatches ** 2):
        W1[:, :, i] = np.random.random((patch_size ** 2, patch_size ** 2)) * 2 * r - r

    W2 = np.zeros((image_size ** 2, image_size ** 2))
    r = np.sqrt(6) / np.sqrt(image_size ** 2 + image_size ** 2 + 1)
    W2[:, :] = np.random.random((image_size ** 2, image_size ** 2)) * 2 * r - r

    theta = np.concatenate((W1.flatten(), W2.flatten()))

    return theta


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

    # theta = np.concatenate((W1.flatten(), W2.flatten(), W3.flatten(), B1.flatten(), B2.flatten(), B3.flatten()))

    theta = np.concatenate((W1.flatten(), W2.flatten(), W3.flatten(), W4.flatten()))

    return theta


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def rectifier(x):
    return np.maximum(0, x)


def rectifier_prime(x):
    return sign(rectifier(x))


def KL_divergence(rho, rho_hat):
    return rho * np.log(rho / rho_hat) + (1.0 - rho) * np.log((1.0 - rho)/(1.0 - rho_hat))


def extract_patches(image, W, nOfPatches, patch_size, image_size):
    z = np.zeros((image_size ** 2))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2] = W[:, :, patchNumber].dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2))
            patchNumber += 1

    return z


def extract_patches_k_sparse(image, W, nOfPatches, patch_size, image_size, K):
    z = np.zeros((image_size ** 2))
    mask_z = np.zeros((image_size ** 2))
    buf = np.zeros((patch_size ** 2))
    mask_buf = np.zeros((patch_size ** 2))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            mask_buf.fill(0)
            buf = W[:, :, patchNumber].dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2))
            z[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2] = buf
            mask_buf[find_K_biggest_indices(buf, K)] = 1
            mask_z[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2] = mask_buf
            patchNumber += 1

    return z, mask_z


def calculate_k_sparsity(image, W, nOfPatches, patch_size, image_size, K):
    z = np.zeros((image_size, image_size))
    mask_z = np.zeros((image_size, image_size))
    buf = np.zeros((patch_size ** 2))
    mask_buf = np.zeros((patch_size ** 2))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            mask_buf.fill(0)
            buf = W[:, :, patchNumber].dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2))
            z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = buf.reshape(patch_size, patch_size)
            mask_buf[find_K_biggest_indices(buf, K)] = 1
            mask_z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = mask_buf.reshape(patch_size, patch_size)
            patchNumber += 1

    return z, mask_z


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


def k_sparsity_bp2(image, nOfPatches, patch_size, image_size, W):
    z = np.zeros((image_size, image_size))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = (W[:, :, patchNumber].T.dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2))).reshape(patch_size, patch_size)
            patchNumber += 1

    return z


def k_sparsity_bp2_N(image, nOfPatches, patch_size, image_size, W, N):
    z = np.zeros((image_size, image_size, N))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :] = (W[:, :, patchNumber].T.dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N))).reshape(patch_size, patch_size, N)
            patchNumber += 1

    return z


def extract_patches_bp(image, nOfPatches, patch_size, image_size):
    z = np.zeros((image_size ** 2))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2] = image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2)
            patchNumber += 1

    return z


def extract_patches_bp2(image, nOfPatches, patch_size, image_size, W):
    z = np.zeros((image_size ** 2))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            z[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2] = W[:, :, patchNumber].T.dot(image[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size].reshape(patch_size ** 2))
            patchNumber += 1

    return z


def combine_patches(image, nOfPatches, patch_size, image_size):
    a = np.zeros((image_size, image_size))

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            a[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = image[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2].reshape(patch_size, patch_size)
            patchNumber += 1

    return a


def find_K_biggest_indices(a, K):
    return np.argsort(a)[::-1][:K]


def find_K_biggest_indices_general(a, K):
    return np.argsort(a, axis=0)[::-1, :][:K, :]


def k_sparse(A, K):
    mask = np.zeros(shape=(A.shape))
    N = A.shape[1]
    # print A.shape

    for i in range(K):
        indexes = np.argmax(A, axis=0)
        for j in range(N):
            # print "indexes: (", j, ", ", indexes[j], ")"
            mask[indexes[j], j] = 1.0
            A[indexes[j], j] = 0

    # print "Sum: ", np.sum(mask, axis=0)

    return mask


def k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images, patch_size, image_size, N, rho_parameter, beta, K):

    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)
    numberOfPatches_l3 = image_size // (4 * patch_size)

    W1 = theta[0:(patch_size ** 4) * (numberOfPatches_l1 ** 2)].reshape(patch_size ** 2, patch_size ** 2, numberOfPatches_l1 ** 2)
    W2 = theta[(patch_size ** 4) * (numberOfPatches_l1 ** 2):(patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)].reshape((2 * patch_size) ** 2, (2 * patch_size) ** 2, numberOfPatches_l2 ** 2)

    start = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)
    end = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2) + ((4 * patch_size) ** 4) * (numberOfPatches_l3 ** 2)

    W3 = theta[start:end].reshape((4 * patch_size) ** 2, (4 * patch_size) ** 2, numberOfPatches_l3 ** 2)
    W4 = theta[end:].reshape(image_size ** 2, image_size ** 2)

    # B1 = theta[end:end + image_size ** 2].reshape(image_size, image_size)
    # B2 = theta[end + image_size ** 2:end + 2 * image_size ** 2].reshape(image_size, image_size)
    # B3 = theta[end + 2 * image_size ** 2:end + 3 * image_size ** 2].reshape(image_size, image_size)

    # Feedforward
    # First hidden layer
    # z2 = np.zeros((image_size, image_size, N))
    # z2_mask = np.zeros((image_size, image_size, N))
    # for i in range(N):
    # for i in range(N):
    #     z2[:, :, i], z2_mask[:, :, i] = calculate_k_sparsity(images[:, :, i], W1, numberOfPatches_l1, patch_size, image_size, K)

    z2, z2_mask = calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

    # z2 += np.repeat(B1[:, :, np.newaxis], N, axis=2)

    z2 *= z2_mask

    # print "Sparsity (first hidden layer): ", np.sum(z2_2D != 0)/N

    # Second hidden layer
    # z3 = np.zeros((image_size, image_size, N))
    # z3_mask = np.zeros((image_size, image_size, N))
    # for i in range(N):
    # for i in range(N):
    #     z3[:, :, i], z3_mask[:, :, i] = calculate_k_sparsity(z2[:, :, i], W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K)

    z3, z3_mask = calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

    # z3 += np.repeat(B2[:, :, np.newaxis], N, axis=2)

    z3 *= z3_mask
    # print "Sparsity (second hidden layer): ", np.sum(z3_2D != 0)/N

    # Third hidden layer
    z4, z4_mask = calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

    z4 *= z4_mask

    z4 = z4.reshape(image_size ** 2, N)
    z4_mask = z4_mask.reshape(image_size ** 2, N)

    # z4 = np.zeros((image_size ** 2, N))
    z5 = W4.dot(z4)

    h = z5.reshape(image_size, image_size, N)

    # h += np.repeat(B3[:, :, np.newaxis], N, axis=2)

    cost = np.sum((h - images) ** 2) / (2 * N)  # + (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W3 ** 2) + np.sum(W2 ** 2) + np.sum(W4 ** 2))

    delta5 = -(images - h).reshape(image_size ** 2, N)

    delta4 = W4.T.dot(delta5)

    delta4 *= z4_mask

    W1_d = np.zeros(shape=(W1.shape))
    W2_d = np.zeros(shape=(W2.shape))
    W3_d = np.zeros(shape=(W3.shape))
    W4_d = np.zeros(shape=(W4.shape))

    # delta2 = np.zeros((image_size, image_size, N))
    delta3 = k_sparsity_bp2_N(delta4.reshape(image_size, image_size, N), numberOfPatches_l3, 4 * patch_size, image_size, W3, N)
    # for i in range(N):
    #     delta2[:, :, i] = k_sparsity_bp2(delta3[:, i].reshape(image_size, image_size), numberOfPatches_l2, 2 * patch_size, image_size, W2)

    delta3 *= z3_mask

    delta2 = k_sparsity_bp2_N(delta3.reshape(image_size, image_size, N), numberOfPatches_l2, 2 * patch_size, image_size, W2, N)
    # for i in range(N):
    #     delta2[:, :, i] = k_sparsity_bp2(delta3[:, i].reshape(image_size, image_size), numberOfPatches_l2, 2 * patch_size, image_size, W2)

    delta2 *= z2_mask

    delta4 = delta4.reshape(image_size, image_size, N)
    patchNumber = 0
    updated_patch_size = (4 * patch_size)
    for x in range(numberOfPatches_l3):
        for y in range(numberOfPatches_l3):
            W3_d[:, :, patchNumber] = (delta4[x * updated_patch_size:(x + 1) * updated_patch_size, y * updated_patch_size:(y + 1) * updated_patch_size, :].reshape(updated_patch_size ** 2, N)).dot(z3[x * updated_patch_size:(x + 1) * updated_patch_size, y * updated_patch_size:(y + 1) * updated_patch_size, :].reshape(updated_patch_size ** 2, N).T)
            patchNumber += 1

    delta3 = delta3.reshape(image_size, image_size, N)
    patchNumber = 0
    for x in range(numberOfPatches_l2):
        for y in range(numberOfPatches_l2):
            W2_d[:, :, patchNumber] = (delta3[x * (2 * patch_size):(x + 1) * (2 * patch_size), y * (2 * patch_size):(y + 1) * (2 * patch_size), :].reshape((2 * patch_size) ** 2, N)).dot(z2[x * (2 * patch_size):(x + 1) * (2 * patch_size), y * (2 * patch_size):(y + 1) * (2 * patch_size), :].reshape((2 * patch_size) ** 2, N).T)
            patchNumber += 1

    patchNumber = 0
    for x in range(numberOfPatches_l1):
        for y in range(numberOfPatches_l1):
            W1_d[:, :, patchNumber] = (delta2[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N)).dot(images[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N).T)
            patchNumber += 1

    W4_d = delta5.dot((z4).reshape(image_size ** 2, N).T) / N  # + lambda_ * W4
    W3_d = W3_d / N  # + lambda_ * W3
    W2_d = W2_d / N  # + lambda_ * W2
    W1_d = W1_d / N  # + lambda_ * W1

    # B1_d = np.sum(delta2, axis=2) / N
    # B2_d = np.sum(delta3, axis=2) / N
    # B3_d = np.sum(delta4.reshape(image_size, image_size, N), axis=2) / N

    # grad = np.concatenate((W1_d.flatten(), W2_d.flatten(), W3_d.flatten(), B1_d.flatten(), B2_d.flatten(), B3_d.flatten()))
    grad = np.concatenate((W1_d.flatten(), W2_d.flatten(), W3_d.flatten(), W4_d.flatten()))

    return cost, grad


def k_sparse_deep_autoencoder_cost(theta, lambda_, images, patch_size, image_size, N, rho_parameter, beta, K):

    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)

    W1 = theta[0:(patch_size ** 4) * (numberOfPatches_l1 ** 2)].reshape(patch_size ** 2, patch_size ** 2, numberOfPatches_l1 ** 2)
    W2 = theta[(patch_size ** 4) * (numberOfPatches_l1 ** 2):(patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)].reshape((2 * patch_size) ** 2, (2 * patch_size) ** 2, numberOfPatches_l2 ** 2)
    W3 = theta[(patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    # First hidden layer
    z2 = np.zeros((image_size ** 2, N))
    z2_mask = np.zeros((image_size ** 2, N))
    for i in range(N):
        z2[:, i], z2_mask[:, i] = extract_patches_k_sparse(images[:, :, i], W1, numberOfPatches_l1, patch_size, image_size, K)

    # Combining all small patches to get the whole image
    z2_mask_2D = np.zeros((image_size, image_size, N))
    z2_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        z2_mask_2D[:, :, i] = combine_patches(z2_mask[:, i], numberOfPatches_l1, patch_size, image_size)
        z2_2D[:, :, i] = combine_patches(z2[:, i], numberOfPatches_l1, patch_size, image_size)

    z2_2D *= z2_mask_2D

    # print "Sparsity (first hidden layer): ", np.sum(z2_2D != 0)/N

    # Second hidden layer
    z3 = np.zeros((image_size ** 2, N))
    z3_mask = np.zeros((image_size ** 2, N))
    for i in range(N):
        z3[:, i], z3_mask[:, i] = extract_patches_k_sparse(z2_2D[:, :, i], W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K)

    # Combining all small patches to get the whole image
    z3_mask_2D = np.zeros((image_size, image_size, N))
    z3_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        z3_mask_2D[:, :, i] = combine_patches(z3_mask[:, i], numberOfPatches_l2, 2 * patch_size, image_size)
        z3_2D[:, :, i] = combine_patches(z3[:, i], numberOfPatches_l2, 2 * patch_size, image_size)

    z3_2D *= z3_mask_2D
    # print "Sparsity (second hidden layer): ", np.sum(z3_2D != 0)/N

    z3_2D = z3_2D.reshape(image_size ** 2, N)
    z3_mask_2D = z3_mask_2D.reshape(image_size ** 2, N)

    z4 = np.zeros((image_size ** 2, N))
    z4 = W3.dot(z3_2D)

    h = z4.reshape(image_size, image_size, N)

    cost = np.sum((h - images) ** 2) / (2 * N) + (lambda_ / 2) * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))

    delta4 = -(images - h).reshape(image_size ** 2, N)

    delta3 = W3.T.dot(delta4)

    delta3 *= z3_mask_2D

    W1_d = np.zeros(shape=(W1.shape))
    W2_d = np.zeros(shape=(W2.shape))
    W3_d = np.zeros(shape=(W3.shape))

    delta3_pathes = np.zeros((image_size ** 2, N))
    delta3_patches_plain = np.zeros((image_size ** 2, N))
    for i in range(N):
        delta3_pathes[:, i] = extract_patches_bp2(delta3[:, i].reshape(image_size, image_size), numberOfPatches_l2, 2 * patch_size, image_size, W2)
        delta3_patches_plain[:, i] = extract_patches_bp(delta3[:, i].reshape(image_size, image_size), numberOfPatches_l2, 2 * patch_size, image_size)

    delta2 = np.zeros((image_size, image_size, N))
    for i in range(N):
        delta2[:, :, i] = combine_patches(delta3_pathes[:, i], numberOfPatches_l2, 2 * patch_size, image_size)

    delta2 *= z2_mask_2D

    delta2_pathes = np.zeros((image_size ** 2, N))
    for i in range(N):
        delta2_pathes[:, i] = extract_patches_bp(delta2[:, :, i], numberOfPatches_l1, patch_size, image_size)

    patchNumber = 0
    for x in range(numberOfPatches_l2):
        for y in range(numberOfPatches_l2):
            W2_d[:, :, patchNumber] = delta3_patches_plain[patchNumber * (2 * patch_size) ** 2:(patchNumber + 1) * (2 * patch_size) ** 2, :].dot(z2_2D[x * (2 * patch_size):(x + 1) * (2 * patch_size), y * (2 * patch_size):(y + 1) * (2 * patch_size), :].reshape((2 * patch_size) ** 2, N).T)
            patchNumber += 1

    patchNumber = 0
    for x in range(numberOfPatches_l1):
        for y in range(numberOfPatches_l1):
            W1_d[:, :, patchNumber] = delta2_pathes[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2, :].dot(images[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N).T)
            patchNumber += 1

    W3_d = delta4.dot((z3_2D).reshape(image_size ** 2, N).T) / N + lambda_ * W3
    W2_d = W2_d / N + lambda_ * W2
    W1_d = W1_d / N + lambda_ * W1

    grad = np.concatenate((W1_d.flatten(), W2_d.flatten(), W3_d.flatten()))

    return cost, grad


def k_sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, N, rho_parameter, beta, K):

    nOfPatches = image_size // patch_size

    W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)
    W2 = theta[(patch_size ** 4) * (nOfPatches ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    z2 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z2[:, i] = extract_patches(images[:, :, i], W1, nOfPatches, patch_size, image_size)

    # a2 = sigmoid(z2)

    # print "K: ", K

    # a2_2D = np.zeros((image_size, image_size, N))
    z2_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        # a2_2D[:, :, i] = combine_patches(a2[:, i], nOfPatches, patch_size, image_size)
        z2_2D[:, :, i] = combine_patches(z2[:, i], nOfPatches, patch_size, image_size)

    z2_2D = z2_2D.reshape(image_size ** 2, N)
    mask = k_sparse(np.copy(z2_2D), K)
    # print mask
    # z2_2D = z2_2D * mask

    z3 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z3[:, i] = W2.dot(z2_2D[:, i] * mask[:, i])

    # h = sigmoid(z3).reshape(image_size, image_size, N)

    h = z3.reshape(image_size, image_size, N)

    # a2_average += np.sum(a2_2D, axis=2)

    # rho_hat = a2_average / N_average

    # rho = np.tile(rho_parameter, rho_hat.shape)

    # eps = np.zeros(shape=(a2_2D.shape))
    # eps.fill(EPSILON)

    cost = np.sum((h - images) ** 2) / (2 * N) + (lambda_ / 2) * (np.sum(W2 ** 2) + np.sum(W1 ** 2))  # + (beta / N) * np.sum(np.sqrt(a2_2D ** 2 + EPSILON)) + lambda_ * np.sum(np.sqrt(W2 ** 2 + EPSILON))  # + (lambda_ / 2) * (np.sum(W2 ** 2))

    # sparsity_prime = a2_2D / np.sqrt(a2_2D ** 2 + EPSILON)

    # print "sparsity_prime.shape: ", sparsity_prime.shape

    delta3 = -(images - h).reshape(image_size ** 2, N)  # * sigmoid_prime(z3)

    # delta2 = (W2.T.dot(delta3) + beta * sparsity_prime.reshape(image_size ** 2, N)) * sigmoid_prime(z2_2D.reshape(image_size ** 2, N))

    delta2 = (W2.T.dot(delta3))  # * sigmoid_prime(z2_2D.reshape(image_size ** 2, N))

    # print "Sum delta2 (before mask): ", np.sum(delta2)
    delta2 = delta2*mask

    # print "Sum delta2 (after mask): ", np.sum(delta2)

    W1_d = np.zeros(shape=(W1.shape))

    delta2_pathes = np.zeros((image_size ** 2, N))
    for i in range(N):
        delta2_pathes[:, i] = extract_patches_bp(delta2[:, i].reshape(image_size, image_size), nOfPatches, patch_size, image_size)

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            W1_d[:, :, patchNumber] = delta2_pathes[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2, :].dot(images[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N).T)
            patchNumber += 1

    W2_d = delta3.dot((z2_2D * mask).reshape(image_size ** 2, N).T) / N + lambda_ * W2  # + lambda_ * W2 / np.sqrt(W2 ** 2 + EPSILON)  # + lambda_ * W2

    W1_d = W1_d / N + lambda_ * W1

    # print "W1_d sum: ", np.sum(W1_d)
    # print "W2_d sum: ", np.sum(W2_d)

    grad = np.concatenate((W1_d.flatten(), W2_d.flatten()))

    return cost, grad


def sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, N, rho_parameter, beta):

    nOfPatches = image_size // patch_size

    W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)
    W2 = theta[(patch_size ** 4) * (nOfPatches ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    z2 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z2[:, i] = extract_patches(images[:, :, i], W1, nOfPatches, patch_size, image_size)

    # a2 = sigmoid(z2)

    # a2_2D = np.zeros((image_size, image_size, N))
    z2_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        # a2_2D[:, :, i] = combine_patches(a2[:, i], nOfPatches, patch_size, image_size)
        z2_2D[:, :, i] = combine_patches(z2[:, i], nOfPatches, patch_size, image_size)

    z3 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z3[:, i] = W2.dot(z2_2D[:, :, i].reshape(image_size ** 2))
        # z3[:, i] = W2.dot(a2_2D[:, :, i].reshape(image_size ** 2))

    # h = sigmoid(z3).reshape(image_size, image_size, N)

    h = z3.reshape(image_size, image_size, N)

    # a2_average += np.sum(a2_2D, axis=2)

    # rho_hat = a2_average / N_average

    # rho = np.tile(rho_parameter, rho_hat.shape)

    # eps = np.zeros(shape=(a2_2D.shape))
    # eps.fill(EPSILON)

    # cost = np.sum((h - images) ** 2) / (2 * N) + (beta / N) * np.sum(np.sqrt(a2_2D ** 2 + EPSILON)) + (lambda_ / 2) * (np.sum(W2 ** 2))
    cost = np.sum((h - images) ** 2) / (2 * N) + (beta / N) * np.sum(np.sqrt(z2_2D ** 2 + EPSILON)) + (lambda_ / 2) * (np.sum(W2 ** 2))

    # sparsity_prime = a2_2D / np.sqrt(a2_2D ** 2 + EPSILON)
    sparsity_prime = z2_2D / np.sqrt(z2_2D ** 2 + EPSILON)

    # print "sparsity_prime.shape: ", sparsity_prime.shape

    delta3 = -(images - h).reshape(image_size ** 2, N)  # * sigmoid_prime(z3)

    delta2 = (W2.T.dot(delta3) + beta * sparsity_prime.reshape(image_size ** 2, N))  # * sigmoid_prime(z2_2D.reshape(image_size ** 2, N))

    # delta2 = (W2.T.dot(delta3)) * sigmoid_prime(z2_2D.reshape(image_size ** 2, N))

    W1_d = np.zeros(shape=(W1.shape))

    delta2_pathes = np.zeros((image_size ** 2, N))
    for i in range(N):
        delta2_pathes[:, i] = extract_patches_bp(delta2[:, i].reshape(image_size, image_size), nOfPatches, patch_size, image_size)

    patchNumber = 0
    for x in range(nOfPatches):
        for y in range(nOfPatches):
            W1_d[:, :, patchNumber] = delta2_pathes[patchNumber * patch_size ** 2:(patchNumber + 1) * patch_size ** 2, :].dot(images[x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size, :].reshape(patch_size ** 2, N).T)
            patchNumber += 1

    W2_d = delta3.dot(z2_2D.reshape(image_size ** 2, N).T) / N + lambda_ * W2  # + lambda_ * W2 / np.sqrt(W2 ** 2 + EPSILON)  # + lambda_ * W2

    W1_d = W1_d / N  # + lambda_ * W1

    grad = np.concatenate((W1_d.flatten(), W2_d.flatten()))

    return cost, grad


def get_average(theta, lambda_, images, patch_size, image_size, N, rho_parameter, beta):

    nOfPatches = image_size // patch_size

    W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)
    W2 = theta[(patch_size ** 4) * (nOfPatches ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    z2 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z2[:, i] = extract_patches(images[:, :, i], W1, nOfPatches, patch_size, image_size)

    a2 = sigmoid(z2)

    a2_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        a2_2D[:, :, i] = combine_patches(a2[:, i], nOfPatches, patch_size, image_size)

    return np.sum(a2_2D, axis=2) / N


def get_representations(theta, images, patch_size, image_size, size):

    nOfPatches = image_size // patch_size

    W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)
    W2 = theta[(patch_size ** 4) * (nOfPatches ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    z2 = np.zeros((image_size ** 2, size))
    for i in range(size):
        z2[:, i] = extract_patches(images[:, :, i], W1, nOfPatches, patch_size, image_size)

    a2 = sigmoid(z2)

    a2_2D = np.zeros((image_size, image_size, size))
    for i in range(size):
        a2_2D[:, :, i] = combine_patches(a2[:, i], nOfPatches, patch_size, image_size)

    z3 = np.zeros((image_size ** 2, size))
    for i in range(size):
        z3[:, i] = W2.dot(a2_2D[:, :, i].reshape(image_size ** 2))

    # h = sigmoid(z3).reshape(image_size, image_size, size)

    h = z3.reshape(image_size, image_size, size)

    print "Representations: "
    for i in range(size):
        subplot(np.sqrt(size), np.sqrt(size), i + 1), imshow(a2_2D[:, :, i], cmap=cm.gray)

    show()

    print "Original images: "
    for i in range(size):
        subplot(np.sqrt(size), np.sqrt(size), i + 1), imshow(images[:, :, i], cmap=cm.gray)

    show()

    print "Output: "
    for i in range(size):
        subplot(np.sqrt(size), np.sqrt(size), i + 1), imshow(h[:, :, i], cmap=cm.gray)

    show()

    subplot(1, 1, 1), title('W2'), imshow(W2[:, :], cmap=cm.gray)

    show()


def get_representations_k_sparse(theta, images, patch_size, image_size, N, K):

    nOfPatches = image_size // patch_size

    W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)
    W2 = theta[(patch_size ** 4) * (nOfPatches ** 2):].reshape(image_size ** 2, image_size ** 2)

    # Feedforward
    z2 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z2[:, i] = extract_patches(images[:, :, i], W1, nOfPatches, patch_size, image_size)

    # a2 = sigmoid(z2)

    # print "K: ", K

    # a2_2D = np.zeros((image_size, image_size, N))
    z2_2D = np.zeros((image_size, image_size, N))
    for i in range(N):
        # a2_2D[:, :, i] = combine_patches(a2[:, i], nOfPatches, patch_size, image_size)
        z2_2D[:, :, i] = combine_patches(z2[:, i], nOfPatches, patch_size, image_size)

    z2_2D = z2_2D.reshape(image_size ** 2, N)
    mask = k_sparse(np.copy(z2_2D), K)
    # print mask

    z3 = np.zeros((image_size ** 2, N))
    for i in range(N):
        z3[:, i] = W2.dot(z2_2D[:, i] * mask[:, i])

    # h = sigmoid(z3).reshape(image_size, image_size, N)

    h = z3.reshape(image_size, image_size, N)

    z2_2D_mask = z2_2D * mask
    z2_2D = z2_2D.reshape(image_size, image_size, N)
    z2_2D_mask = z2_2D_mask.reshape(image_size, image_size, N)

    print "Representations (without mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(z2_2D[:, :, i], cmap=cm.gray)

    show()

    print "Representations (with mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow((z2_2D_mask)[:, :, i], cmap=cm.gray)

    show()

    print "Original images: "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(images[:, :, i], cmap=cm.gray)

    show()

    print "Output: "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(h[:, :, i], cmap=cm.gray)

    show()

    subplot(1, 1, 1), title('W2'), imshow(W2[:, :], cmap=cm.gray)

    show()


def get_representations_k_deep_sparse(theta, images, patch_size, image_size, N, K):

    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)
    numberOfPatches_l3 = image_size // (4 * patch_size)

    W1 = theta[0:(patch_size ** 4) * (numberOfPatches_l1 ** 2)].reshape(patch_size ** 2, patch_size ** 2, numberOfPatches_l1 ** 2)
    W2 = theta[(patch_size ** 4) * (numberOfPatches_l1 ** 2):(patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)].reshape((2 * patch_size) ** 2, (2 * patch_size) ** 2, numberOfPatches_l2 ** 2)

    start = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)
    end = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2) + ((4 * patch_size) ** 4) * (numberOfPatches_l3 ** 2)

    W3 = theta[start:end].reshape((4 * patch_size) ** 2, (4 * patch_size) ** 2, numberOfPatches_l3 ** 2)
    W4 = theta[end:].reshape(image_size ** 2, image_size ** 2)

    # B1 = theta[end:end + image_size ** 2].reshape(image_size, image_size)
    # B2 = theta[end + image_size ** 2:end + 2 * image_size ** 2].reshape(image_size, image_size)
    # B3 = theta[end + 2 * image_size ** 2:end + 3 * image_size ** 2].reshape(image_size, image_size)

    # Feedforward
    # First hidden layer

    z2, z2_mask = calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

    # z2 += np.repeat(B1[:, :, np.newaxis], N, axis=2)

    print "Representations hl1 (without mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(z2[:, :, i], cmap=cm.gray)

    show()

    z2 *= z2_mask

    print "Representations hl1 (with mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow((z2)[:, :, i], cmap=cm.gray)

    show()

    # print "Sparsity (first hidden layer): ", np.sum(z2_2D != 0)/N

    # Second hidden layer

    z3, z3_mask = calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

    # z3 += np.repeat(B2[:, :, np.newaxis], N, axis=2)

    print "Representations hl2 (without mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(z3[:, :, i], cmap=cm.gray)

    show()

    z3 *= z3_mask

    print "Representations hl2 (with mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow((z3)[:, :, i], cmap=cm.gray)

    show()

    # Third hidden layer
    z4, z4_mask = calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

    print "Representations hl3 (without mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(z4[:, :, i], cmap=cm.gray)

    show()

    z4 *= z4_mask

    print "Representations hl3 (with mask): "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(z4[:, :, i], cmap=cm.gray)

    show()

    z4 = z4.reshape(image_size ** 2, N)
    z4_mask = z4_mask.reshape(image_size ** 2, N)

    # z4 = np.zeros((image_size ** 2, N))
    z5 = W4.dot(z4)

    h = z5.reshape(image_size, image_size, N)

    print "Original images: "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(images[:, :, i], cmap=cm.gray)

    show()

    print "Output: "
    for i in range(N):
        subplot(np.sqrt(N), np.sqrt(N), i + 1), imshow(h[:, :, i], cmap=cm.gray)

    show()

    print "W1: "
    k = 0
    for i in range(numberOfPatches_l1 ** 2):
        for j in range(patch_size ** 2):
            subplot(patch_size * numberOfPatches_l1, patch_size * numberOfPatches_l1, k + 1), imshow(W1[:, j, i].reshape(patch_size, patch_size), cmap=cm.gray)
            k += 1
    show()

    # print "W1 (inverse): "
    # k = 0
    # for i in range(numberOfPatches_l1 ** 2):
    #     for j in range(patch_size ** 2):
    #         subplot(patch_size * numberOfPatches_l1, patch_size * numberOfPatches_l1, k + 1), imshow(W1[j, :, i].reshape(patch_size, patch_size), cmap=cm.gray)
    #         k += 1
    # show()

    print "W2: "
    k = 0
    for i in range(numberOfPatches_l2 ** 2):
        for j in range((2 * patch_size) ** 2):
            subplot((2 * patch_size) * numberOfPatches_l2, (2 * patch_size) * numberOfPatches_l2, k + 1), imshow(W2[:, j, i].reshape(2 * patch_size, 2 * patch_size), cmap=cm.gray)
            k += 1
    show()

    print "W3: "
    k = 0
    updated_patch_size = (4 * patch_size)
    for i in range(numberOfPatches_l3 ** 2):
        for j in range(updated_patch_size ** 2):
            subplot(updated_patch_size * numberOfPatches_l3, updated_patch_size * numberOfPatches_l3, k + 1), imshow(W3[:, j, i].reshape(updated_patch_size, updated_patch_size), cmap=cm.gray)
            k += 1
    show()

    # print "W2 (inverse): "
    # k = 0
    # for i in range(numberOfPatches_l2 ** 2):
    #     for j in range((2 * patch_size) ** 2):
    #         subplot((2 * patch_size) * numberOfPatches_l2, (2 * patch_size) * numberOfPatches_l2, k + 1), imshow(W2[j, :, i].reshape(2 * patch_size, 2 * patch_size), cmap=cm.gray)
    #         k += 1
    # show()

    print "W4: "
    subplot(1, 1, 1), title('W4'), imshow(W4[:, :], cmap=cm.gray)

    show()


def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):

    m = data.shape[1]
    labels[labels == 10] = 0
    # print 'm: ', m
    theta = theta.reshape(num_classes, input_size)
    # print "theta.shape: ", theta.shape
    # print "data.hsape: ", data.shape
    theta_data = theta.dot(data)
    theta_data = theta_data - np.max(theta_data)
    # print "theta_data.shape: ", theta_data.shape
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    # print "prob_data.shape: ", prob_data.shape
    # print "labels.shape: ", labels.shape
    # print "type(labels): ", type(labels)
    # print "np.unique(labels): ", np.unique(labels)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    # print "indicator.shape: ", indicator.shape
    indicator = np.array(indicator.todense())
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lambda_ / 2) * np.sum(theta * theta)
    # print "(indicator - prob_data).dot(data.transpose()).shape:", ((indicator - prob_data).dot(data.transpose())).shape
    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta

    return cost, grad.flatten()


def softmax_train(input_size, num_classes, lambda_, data, labels, options={'maxiter': 400, 'disp': True}):
    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)

    J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

    print result
    # Return optimum theta, input size & num classes
    opt_theta = result.x

    return opt_theta, input_size, num_classes


def softmax_predict(model, data):
    opt_theta, input_size, num_classes = model
    opt_theta = opt_theta.reshape(num_classes, input_size)

    prod = opt_theta.dot(data)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)

    return pred


def pickle_data(file_name, data):
    f = open(file_name, 'wb')
    cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)
    f.close()


def unpickle_data(file_name):
    f = open(file_name, 'rb')
    data = cPickle.load(f)
    f.close()
    return data


def run_softmax():
    image_size = 32
    patch_size = 4

    # open training data
    print "Training data!"
    path = 'data/train_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_train.pickle"
    file_name_labels = "data/pickles/labels_train.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Training data were loaded and whitened!"

    images_train = unpickle_data(file_name_data)[:, :, :N]
    labels_train = unpickle_data(file_name_labels)[:N]

    # open test data
    print "Test data!"
    path = 'data/test_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_test.pickle"
    file_name_labels = "data/pickles/labels_test.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Test data were loaded and whitened!"

    images_test = unpickle_data(file_name_data)
    labels_test = unpickle_data(file_name_labels)

    # open saved theta parameters
    theta = np.load('weights_learned/weights.out.npy')

    # get representations for training data
    file_name = "data/pickles/training_features.pickle"
    if not os.path.isfile(file_name):
        train_features = get_represenations(theta, images_train, patch_size, image_size, images_train.shape[2], 2)
        pickle_data(file_name, train_features)
        print "Pickled training features!"
    else:
        train_features = unpickle_data(file_name)
        print "Unpickled training features!"

    # get representations for test data
    file_name = "data/pickles/test_features.pickle"
    if not os.path.isfile(file_name):
        test_features = get_represenations(theta, images_test, patch_size, image_size, images_test.shape[2], 2)
        pickle_data(file_name, test_features)
        print "Pickled testing features!"
    else:
        test_features = unpickle_data(file_name)
        print "Unpickled testing features!"

    # print "Check gradients!"
    # lambda_ = 0.1
    # num_labels = 10
    # theta = 0.005 * np.random.randn(num_labels * image_size ** 2)
    # l_cost, l_grad = softmax_cost(theta, num_labels, image_size ** 2, lambda_, train_features, labels_train)
    # J = lambda x: softmax_cost(x, num_labels, image_size ** 2, lambda_, train_features, labels_train)
    # gradient_check.compute_grad(J, theta, l_grad)

    # run softmax function
    lambda_ = 0.01
    # lambda_ = 0.1
    options_ = {'maxiter': 500, 'disp': True}
    num_labels = 10

    opt_theta, input_size, num_classes = softmax_train(image_size ** 2, num_labels, lambda_, train_features, labels_train, options_)

    predictions = softmax_predict((opt_theta, image_size ** 2, num_labels), test_features)
    labels_test[labels_test == 10] = 0
    print "Accuracy: {:.2%}".format(np.sum(predictions == labels_test, dtype=np.float64) / labels_test.shape[0])


def get_represenations(theta, images, patch_size, image_size, N, K):

    numberOfPatches_l1 = image_size // patch_size
    numberOfPatches_l2 = image_size // (2 * patch_size)
    numberOfPatches_l3 = image_size // (4 * patch_size)

    W1 = theta[0:(patch_size ** 4) * (numberOfPatches_l1 ** 2)].reshape(patch_size ** 2, patch_size ** 2, numberOfPatches_l1 ** 2)
    W2 = theta[(patch_size ** 4) * (numberOfPatches_l1 ** 2):(patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)].reshape((2 * patch_size) ** 2, (2 * patch_size) ** 2, numberOfPatches_l2 ** 2)

    start = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2)
    end = (patch_size ** 4) * (numberOfPatches_l1 ** 2) + ((2 * patch_size) ** 4) * (numberOfPatches_l2 ** 2) + ((4 * patch_size) ** 4) * (numberOfPatches_l3 ** 2)

    W3 = theta[start:end].reshape((4 * patch_size) ** 2, (4 * patch_size) ** 2, numberOfPatches_l3 ** 2)
    W4 = theta[end:].reshape(image_size ** 2, image_size ** 2)

    # B1 = theta[end:end + image_size ** 2].reshape(image_size, image_size)
    # B2 = theta[end + image_size ** 2:end + 2 * image_size ** 2].reshape(image_size, image_size)
    # B3 = theta[end + 2 * image_size ** 2:end + 3 * image_size ** 2].reshape(image_size, image_size)

    # Feedforward
    # First hidden layer
    # z2 = np.zeros((image_size, image_size, N))
    # z2_mask = np.zeros((image_size, image_size, N))
    # for i in range(N):
    # for i in range(N):
    #     z2[:, :, i], z2_mask[:, :, i] = calculate_k_sparsity(images[:, :, i], W1, numberOfPatches_l1, patch_size, image_size, K)

    z2, z2_mask = calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

    # z2 += np.repeat(B1[:, :, np.newaxis], N, axis=2)

    z2 *= z2_mask

    print "Done first hidden layer!"

    # print "Sparsity (first hidden layer): ", np.sum(z2_2D != 0)/N

    # Second hidden layer
    # z3 = np.zeros((image_size, image_size, N))
    # z3_mask = np.zeros((image_size, image_size, N))
    # for i in range(N):
    # for i in range(N):
    #     z3[:, :, i], z3_mask[:, :, i] = calculate_k_sparsity(z2[:, :, i], W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K)

    z3, z3_mask = calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

    # z3 += np.repeat(B2[:, :, np.newaxis], N, axis=2)

    z3 *= z3_mask

    print "Done second hidden layer!"

    # print "Sparsity (second hidden layer): ", np.sum(z3_2D != 0)/N

    # Third hidden layer
    z4, z4_mask = calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

    z4 *= z4_mask

    print "Done third hidden layer!"

    return z4.reshape(image_size ** 2, N)


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


def prepare_data():
    print "Started reading file."
    data = scipy.io.loadmat('data/train_32x32.mat')
    print "Complete reading file."

    X_original = data['X']
    y = data['y']

    print "y.shape: ", y.shape

    row, col, rgb, m = X_original.shape
    print row, col
    X = np.zeros((m, row, col))

    # converting RGB into grayscale format
    for index in range(m):
        r, g, b = X_original[:, :, 0, index], \
            X_original[:, :, 1, index], X_original[:, :, 2, index]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        X[index, :, :] = gray
        # X[index, :] = gray.flatten()

    X /= 255.0

    # X = whitening(X.T[:, :, :N+36]).T
    X = whitening(X.T).T

    print "Max element: ", np.max(X)
    print "Min element: ", np.min(X)

    # Min, Max = np.min(X), np.max(X)
    # X = (X - Min)/(Max - Min)

    # print "Max element (after): ", np.max(X)
    # print "Min element (after): ", np.min(X)

    # X_sub = X.T[:, :, :100]
    # for i in range(10 ** 2):
    #     subplot(10, 10, i + 1), imshow(X_sub[:, :, i], cmap=cm.gray)
    # show()

    # N = 1000
    print "X.shape: ", X.shape
    # images = X.T
    images = X.T[:, :, :N]
    y = y[:N, 0]
    images_repr = X.T[:, :, N:N+36]
    # print "images.shape: ", images.shape
    # print "images: ", images

    # for i in range(10):
    #     # print y == (i + 1)
    #     images_sub = images[:, :, y == (i + 1)]
    #     for j in range(16):
    #         subplot(np.sqrt(16), np.sqrt(16), j + 1), imshow(images_sub[:, :, j], cmap=cm.gray)

    #     show()

    # whitening(images)

    return images, y, images_repr


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


def test():
    print "Started reading file."
    data = scipy.io.loadmat('data/test_32x32.mat')
    print "Complete reading file."

    X_original = data['X']
    y = data['y']

    row, col, rgb, m = X_original.shape
    print row, col, m
    X = np.zeros((m, row, col))

    # converting RGB into grayscale format
    for index in range(m):
        r, g, b = X_original[:, :, 0, index], \
            X_original[:, :, 1, index], X_original[:, :, 2, index]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        X[index, :, :] = gray
    # X[index, :] = gray.flatten()

    X /= 255.0

    # N = 73200
    # print "X.shape: ", X.shape
    images_repr = X.T[:, :, N:N+36]
    images = X.TT[:, :, N:N+36]
    y = y[:, 0]

    image_size = 32
    patch_size = 4

    # for i in range(10):
    # print y == (i + 1)
    images_sub = images[:, :, y == (4 + 1)]
    images_sub = images_sub[:, :, :36]
    theta0 = np.load('weights' + str(1) + '.out.npy')

    nOfPatches = image_size // patch_size
    # W1 = theta0[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)

    # for i in range(nOfPatches ** 2):
    #     subplot(nOfPatches, nOfPatches, i + 1), title('W1'), imshow(W1[:, :, i], cmap=cm.gray)
    # show()

    get_representations(theta0, images_sub, patch_size, image_size, images_sub.shape[2])


def run_sparse_autoencoder():
    image_size = 32
    patch_size = 4
    # num_labels = 10
    lambda_ = 0.001
    rho = 0.05
    nju = 3e-3
    beta = 0.5
    mu = 0.9

    # images_all, y, images_repr = prepare_data()

    # open training data
    print "Trainig data!"
    path = 'data/train_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_train.pickle"
    file_name_labels = "data/pickles/labels_train.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Training data were loaded and whitened!"

    images_all = unpickle_data(file_name_data)[:, :, :N]
    images_repr = unpickle_data(file_name_data)[:, :, N:N+36]
    y = unpickle_data(file_name_labels)[:N]

    # theta = initialize(patch_size, image_size)
    theta = initialize_k_deep_sparse_autoencoder(patch_size, image_size)

    # print "Check gradients!"
    # lambda_ = 0.1
    # l_cost, l_grad = k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images_all, patch_size, image_size, N, rho, beta, 2)
    # J = lambda x: k_sparse_deep_autoencoder_cost_without_patches(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 2)
    # gradient_check.compute_grad(J, theta, l_grad)

    # print k_sparse_deep_autoencoder_cost(theta, lambda_, images_all, patch_size, image_size, N, rho, beta, patch_size)

    # J = lambda x: k_sparse_deep_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 1)
    J = lambda x: k_sparse_deep_autoencoder_cost_without_patches(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 2)
    # # # J = lambda x: k_sparse_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 200)
    # # # J = lambda x: sparse_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta)
    options_ = {'maxiter': 200, 'disp': True}
    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    print theta.shape
    theta = result.x

    # a2_average = np.zeros((image_size, image_size))

    # max_iter = 100
    # batch_size = 1000
    # n_batches = N // batch_size
    # print "n_batches: ", n_batches
    # learning_rate0 = 0.01
    # learning_rate = learning_rate0
    # iter = 0
    # v = 0
    # while iter < max_iter:
    #     iter += 1
    #     s = 0
    #     # N_average = 0
    #     # a2_average = get_average(theta, lambda_, images, patch_size, image_size, N, rho, beta)
    #     # a2_average.fill(0)
    #     for b in range(n_batches):
    #         batch_begin = b * batch_size
    #         N_average = (b + 1) * batch_size
    #         batch_end = batch_begin + batch_size
    #         X_batch = images_all[:, :, batch_begin:batch_end]
    #         cost, grad = k_sparse_deep_autoencoder_cost(theta, lambda_, X_batch, patch_size, image_size, batch_size, rho, beta, patch_size)
    #         # theta -= learning_rate * grad

    #         # momentum
    #         v = mu * v - learning_rate * grad
    #         theta += v

    #         # J = lambda x: sparse_autoencoder_cost(x, lambda_, X_batch, patch_size, image_size, batch_size, rho, beta, a2_average)
    #         # options_ = {'maxiter': 1, 'disp': True}
    #         # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    #         # theta = result.x
    #         s += cost
    #         print "Cost: %s, batch: %d", (cost, b)
    #     print "Cost: %s, iter: %d", (s/n_batches, iter), "\n"
    #     print "learning_rate: %d", (learning_rate), "\n"
    #     # learning_rate = learning_rate0 / (1 + iter / float(batch_size))

    # for i in range(10):
    #     # print y == (i + 1)
    #     print 'weights' + str(i + 1) + '.out'
    #     images = images_all[:, :, y == (i + 1)]
    #     theta = initialize(patch_size, image_size)
    #     N = images.shape[2]
    #     max_iter = 40
    #     batch_size = 1000
    #     n_batches = N // batch_size
    #     print "n_batches: ", n_batches
    #     learning_rate0 = 0.09
    #     learning_rate = learning_rate0
    #     iter = 0
    #     while iter < max_iter:
    #         iter += 1
    #         s = 0
    #         N_average = 0
    #         # a2_average = get_average(theta, lambda_, images, patch_size, image_size, N, rho, beta)
    #         a2_average.fill(0)
    #         for b in range(n_batches):
    #             batch_begin = b * batch_size
    #             N_average = (b + 1) * batch_size
    #             batch_end = batch_begin + batch_size
    #             X_batch = images[:, :, batch_begin:batch_end]
    #             cost, grad, a2_average = sparse_autoencoder_cost(theta, lambda_, X_batch, patch_size, image_size, batch_size, rho, beta, a2_average, N_average)
    #             theta -= learning_rate * grad
    #             # J = lambda x: sparse_autoencoder_cost(x, lambda_, X_batch, patch_size, image_size, batch_size, rho, beta, a2_average)
    #             # options_ = {'maxiter': 1, 'disp': True}
    #             # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    #             # theta = result.x
    #             s += cost
    #             print "Cost: %s, batch: %d", (cost, b)
    #         print "Cost: %s, iter: %d", (s/n_batches, iter), "\n"
    #         print "learning_rate: %d", (learning_rate), "\n"
    #         learning_rate = learning_rate0 / (1 + iter / float(batch_size))
    # np.save('weights' + str(i + 1) + '.out', theta)

    # l_cost, l_grad = deep_autoencoder_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, N)

    # for i in range(2):
    #     l_cost, l_grad = deep_autoencoder_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, N)
    #     print "Cost: %s, iteration: %d", (l_cost, i)
    #     theta = theta - 0.1*l_grad

    np.save('weights_learned/weights.out', theta)

    theta = np.load('weights_learned/weights.out.npy')

    # nOfPatches = image_size // patch_size
    # W1 = theta[0:(patch_size ** 4) * (nOfPatches ** 2)].reshape(patch_size ** 2, patch_size ** 2, nOfPatches ** 2)

    # for i in range(nOfPatches ** 2):
    #     subplot(nOfPatches, nOfPatches, i + 1), imshow(W1[:, :, i], cmap=cm.gray)

    # # # # print "--- %s seconds ---" % (time.time() - start_time)

    # show()

    # get_representations_k_sparse(theta, images_repr, patch_size, image_size, images_repr.shape[2], 200)
    # get_representations(theta, images_repr, patch_size, image_size, images_repr.shape[2])
    get_representations_k_deep_sparse(theta, images_repr, patch_size, image_size, images_repr.shape[2], 2)

run_sparse_autoencoder()

# print timeit.timeit("run_sparse_autoencoder()", "from __main__ import run_sparse_autoencoder", number=1)

# run_softmax()

# test()
