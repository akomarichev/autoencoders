import numpy as np
import os.path
import scipy.optimize

import helper
import initialization as init
import model
import representations as rep
import gradient_check
import preprocessing as prepr


def run_sparse_autoencoder(N, image_size, patch_size, prepare_data=True):
    # images_all, y, images_repr = prepare_data()

    # open training data
    print "Trainig data!"
    train_path = 'data/train_32x32.mat'
    test_path = 'data/test_32x32.mat'
    file_train = "data/pickles/train.pickle"
    file_val = "data/pickles/val.pickle"

    if prepare_data:
        X, y = prepr.load_data(train_path)
        X_test, y_test = prepr.load_data(test_path)
        prepr.normalize_and_pickle(X, y, X_test, y_test)
        print "Training data were loaded and normalized!"

    images_train = helper.unpickle_data(file_train)[:, :, :N]
    images_val = helper.unpickle_data(file_val)
    images_repr = images_val[:, :, :36]

    theta = init.initialize_k_deep_sparse_autoencoder(patch_size, image_size)

    max_iter = 200
    batch_size = 1000
    n_batches = N // batch_size
    print "n_batches: ", n_batches
    learning_rate = 1e-3
    learning_rate_decay = 0.95
    mu = 0.9
    lambda_ = 0.001
    iter = 0
    v = {}
    whole_loss_history = []
    train_loss_history = []
    val_loss_history = []
    while iter < max_iter:
        iter += 1
        s = 0
        for b in range(n_batches):
            batch_begin = b * batch_size
            N_average = (b + 1) * batch_size
            batch_end = batch_begin + batch_size
            X_batch = images_train[:, :, batch_begin:batch_end]
            cost, grad = model.k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, X_batch, patch_size, image_size, batch_size, patch_size)
            whole_loss_history.append(cost)

            # momentum update
            for item in grad:
                if item not in v:
                    v[item] = np.zeros(grad[item].shape)
                v[item] = mu * v[item] - learning_rate * grad[item]
                theta[item] += v[item]

        mask = np.random.choice(N, 1000)
        train_subset = images_train[:, :, mask]
        cost_train = model.k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, train_subset, patch_size, image_size, 1000, patch_size)[0]
        train_loss_history.append(cost_train)
        cost_val = model.k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images_val, patch_size, image_size, images_val.shape[2], patch_size)[0]
        train_loss_history.append(cost_val)
        print "Cost_train: ", cost_train, ", cost_val: ", cost_val, ", epoch: ", iter, " learning_rate: %d", (learning_rate)
        learning_rate *= learning_rate_decay

    # print "Check gradients!"
    # lambda_ = 0.1
    # l_cost, l_grad = model.k_sparse_deep_autoencoder_cost_without_patches(theta, lambda_, images_train, patch_size, image_size, N, 2)
    # # helper.check_sparsity_of_gradients(l_grad, 'W3')
    # J = lambda x: model.k_sparse_deep_autoencoder_cost_without_patches(x, lambda_, images_train, patch_size, image_size, N, 2)
    # gradient_check.compute_grad(J, theta, l_grad)

    helper.pickle_data('weights_learned/weights.out', theta)
    helper.pickle_data('loss_history/whole_loss_history.data', whole_loss_history)
    helper.pickle_data('loss_history/train_loss_history.data', train_loss_history)
    helper.pickle_data('loss_history/val_loss_history.data', val_loss_history)
    theta = helper.unpickle_data('weights_learned/weights.out')

    rep.get_representations_k_deep_sparse(theta, images_repr, patch_size, image_size, images_repr.shape[2], 2)
