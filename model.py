import numpy as np

import helper


def k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images, patch_size, image_size, N, K):

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

    z2, z2_mask = helper.calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

    # z2 += np.repeat(B1[:, :, np.newaxis], N, axis=2)

    z2 *= z2_mask

    # print "Sparsity (first hidden layer): ", np.sum(z2_2D != 0)/N

    # Second hidden layer
    # z3 = np.zeros((image_size, image_size, N))
    # z3_mask = np.zeros((image_size, image_size, N))
    # for i in range(N):
    # for i in range(N):
    #     z3[:, :, i], z3_mask[:, :, i] = calculate_k_sparsity(z2[:, :, i], W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K)

    z3, z3_mask = helper.calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

    # z3 += np.repeat(B2[:, :, np.newaxis], N, axis=2)

    z3 *= z3_mask
    # print "Sparsity (second hidden layer): ", np.sum(z3_2D != 0)/N

    # Third hidden layer
    z4, z4_mask = helper.calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

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
    delta3 = helper.k_sparsity_bp2_N(delta4.reshape(image_size, image_size, N), numberOfPatches_l3, 4 * patch_size, image_size, W3, N)
    # for i in range(N):
    #     delta2[:, :, i] = k_sparsity_bp2(delta3[:, i].reshape(image_size, image_size), numberOfPatches_l2, 2 * patch_size, image_size, W2)

    delta3 *= z3_mask

    delta2 = helper.k_sparsity_bp2_N(delta3.reshape(image_size, image_size, N), numberOfPatches_l2, 2 * patch_size, image_size, W2, N)
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


def feed_forward(theta, images, patch_size, image_size, N, K):

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

    z2, z2_mask = helper.calculate_k_sparsity_N(images, W1, numberOfPatches_l1, patch_size, image_size, K, N)

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

    z3, z3_mask = helper.calculate_k_sparsity_N(z2, W2, numberOfPatches_l2, 2 * patch_size, image_size, 2 * K, N)

    # z3 += np.repeat(B2[:, :, np.newaxis], N, axis=2)

    z3 *= z3_mask

    print "Done second hidden layer!"

    # print "Sparsity (second hidden layer): ", np.sum(z3_2D != 0)/N

    # Third hidden layer
    z4, z4_mask = helper.calculate_k_sparsity_N(z3, W3, numberOfPatches_l3, 4 * patch_size, image_size, 4 * K, N)

    z4 *= z4_mask

    print "Done third hidden layer!"

    return z4.reshape(image_size ** 2, N)
