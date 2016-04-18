import numpy as np
import os.path
import scipy.optimize

import helper
import initialization as init
import model
import representations as rep

# some constants
mu = 0.9
lambda_ = 0.001


def run_sparse_autoencoder(N, image_size, patch_size):
    # images_all, y, images_repr = prepare_data()

    # open training data
    print "Trainig data!"
    path = 'data/train_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_train.pickle"
    file_name_labels = "data/pickles/labels_train.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Training data were loaded and whitened!"

    images_all = helper.unpickle_data(file_name_data)[:, :, :N]
    images_repr = helper.unpickle_data(file_name_data)[:, :, N:N+36]
    y = helper.unpickle_data(file_name_labels)[:N]

    # theta = initialize(patch_size, image_size)
    theta = init.initialize_k_deep_sparse_autoencoder(patch_size, image_size)

    # print "Check gradients!"
    # lambda_ = 0.1
    # l_cost, l_grad = k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images_all, patch_size, image_size, N, rho, beta, 2)
    # J = lambda x: k_sparse_deep_autoencoder_cost_without_patches(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 2)
    # gradient_check.compute_grad(J, theta, l_grad)

    # print k_sparse_deep_autoencoder_cost(theta, lambda_, images_all, patch_size, image_size, N, rho, beta, patch_size)

    # J = lambda x: k_sparse_deep_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 1)
    J = lambda x: model.k_sparse_deep_autoencoder_cost_without_patches(x, lambda_, images_all, patch_size, image_size, N, 2)
    # # # J = lambda x: k_sparse_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta, 200)
    # # # J = lambda x: sparse_autoencoder_cost(x, lambda_, images_all, patch_size, image_size, N, rho, beta)
    options_ = {'maxiter': 2, 'disp': True}
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
    rep.get_representations_k_deep_sparse(theta, images_repr, patch_size, image_size, images_repr.shape[2], 2)
