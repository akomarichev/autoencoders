import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import matplotlib

import helper


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

    z2, z2_mask = helper.calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

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

    z3, z3_mask = helper.calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

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
    z4, z4_mask = helper.calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

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
