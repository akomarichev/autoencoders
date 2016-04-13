import numpy as np
# import deep_autoencoder
# import sparse_autoencoder
# import convolution_layer
# import conv_maxout
# import conv_maxout_log_regression
# import sparseAutoencoder
# # import sparse_smoothed_autoencoder
# import scipy.optimize

patch_size = 8
small_patch = 4
image_size = 32


def compute_grad(J, theta, l_grad):
    EPSILON = 0.0001

    grad = np.zeros(theta.shape, dtype=np.float64)

    print "grad.shape: ", grad.shape
    print "theta.shape: ", theta.shape

    dim2 = ((image_size - patch_size + 1) - patch_size + 1) * ((image_size - patch_size + 1) - patch_size + 1)
    # start = 2 ** 4 * 16 ** 2 + 4 ** 4 * 8 ** 2
    # start = 0
    start = 4 ** 4 * 8 ** 2  # + 8 ** 4 * 4 ** 2
    # start = 8 ** 4 * 4 ** 2
    # start = patch_size * patch_size  # + dim2 * small_patch ** 2

    # theta.shape[0]
    # for i in range(theta.shape[0]):
    for i in range(start, start + 1001):
        # for i in range(start, theta.shape[0]):
        theta_epsilon_plus = np.array(theta, dtype=np.float64)
        theta_epsilon_plus[i] = theta[i] + EPSILON
        theta_epsilon_minus = np.array(theta, dtype=np.float64)
        theta_epsilon_minus[i] = theta[i] - EPSILON
        # print "J(theta_epsilon_plus): ", J(theta_epsilon_plus)[0]
        # print "J(theta_epsilon_minus): ", J(theta_epsilon_minus)[0]
        # print "(J(theta_epsilon_plus) - J(theta_epsilon_minus)): ", (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0])
        # print "(J(theta_epsilon_plus) - J(theta_epsilon_minus)) / (2 * EPSILON): ", (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * EPSILON)
        grad[i] = (J(theta_epsilon_plus)[0] - J(theta_epsilon_minus)[0]) / (2 * EPSILON)
        if i % 50 == 0 and i != 0:
            print "Computing gradient for input:", i
            # print "Diff: ", np.linalg.norm(grad[start:i] - l_grad[start:i]) / np.linalg.norm(grad[start:i] + l_grad[start:i])
            # print "l_grad[:i]: ", l_grad[start:i]
            # print "grad[:i]: ", grad[start:i]
            print "Diff: ", np.linalg.norm(grad[start:i] - l_grad[start:i]) / np.linalg.norm(grad[start:i] + l_grad[start:i])
            print "L1 norm: ", np.linalg.norm(grad[start:i], 1)
            print "l_grad[:i]: ", l_grad[start:i]
            print "grad[:i]: ", grad[start:i]
    print "grad[:20]: ", grad

    print "Diff: ", np.linalg.norm(grad - l_grad) / np.linalg.norm(grad + l_grad)
    print "L1 norm: ", np.linalg.norm(grad, 1)
    return grad


def check_grad_in_deep_autoencoder():
    image_size = 32
    patch_size = 9
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = deep_autoencoder.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1) * (image_size - patch_size + 1)

    theta = deep_autoencoder.initialize(hidden_size, visible_size, patch_size)

    pooled_size = (image_size - patch_size + 1)/deep_autoencoder.pool_size

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    l_cost, l_grad = deep_autoencoder.deep_autoencoder_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)

    J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad[:20]


def check_grad_in_sparse_smoothed_autoencoder():
    image_size = 32
    patch_size = 4
    # num_labels = 10
    lambda_ = 1e-2
    rho = 0.05
    nju = 3e-3
    beta = 0.5
    mu = 0.9

    images = sparse_smoothed_autoencoder.prepare_data()

    theta = sparse_smoothed_autoencoder.initialize(patch_size, image_size)

    # print "cost: ", sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, N, rho, beta)

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # l_cost, l_grad = sparse_smoothed_autoencoder.k_sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta, 100)
    l_cost, l_grad = sparse_smoothed_autoencoder.sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta)

    # print "L_cost: ", l_cost
    # J = lambda x: sparse_smoothed_autoencoder.k_sparse_autoencoder_cost(x, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta, 100)
    J = lambda x: sparse_smoothed_autoencoder.sparse_autoencoder_cost(x, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_k_deep_sparse_autoencoder():
    image_size = 32
    patch_size = 2
    # num_labels = 10
    lambda_ = 1e-2
    rho = 0.05
    nju = 3e-3
    beta = 0.5
    mu = 0.9

    images = sparse_smoothed_autoencoder.prepare_data()

    theta = sparse_smoothed_autoencoder.initialize_k_deep_sparse_autoencoder(patch_size, image_size)

    l_cost, l_grad = sparse_smoothed_autoencoder.k_sparse_deep_autoencoder_cost(theta, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta, patch_size)

    J = lambda x: sparse_smoothed_autoencoder.k_sparse_deep_autoencoder_cost(x, lambda_, images, patch_size, image_size, sparse_smoothed_autoencoder.N, rho, beta, patch_size)

    computed_grad = compute_grad(J, theta, l_grad)


def check_grad_in_sparse_autoencoder():
    image_size = 32
    patch_size = 4
    # num_labels = 10
    lambda_ = 0.1
    rho = 0.1
    nju = 3e-3
    beta = 2

    images = sparseAutoencoder.prepare_data()

    theta = sparseAutoencoder.initialize(patch_size, image_size)

    # print "cost: ", sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, N, rho, beta)

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    l_cost, l_grad = sparseAutoencoder.sparse_autoencoder_cost(theta, lambda_, images, patch_size, image_size, sparseAutoencoder.N, rho, beta)

    J = lambda x: sparseAutoencoder.sparse_autoencoder_cost(x, lambda_, images, patch_size, image_size, sparseAutoencoder.N, rho, beta)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_convolution_layer():
    image_size = 32
    patch_size = 9
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1) * (image_size - patch_size + 1)

    theta = convolution_layer.initialize(hidden_size, visible_size, patch_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad[:20]


def check_grad_in_two_convolution_layers():
    image_size = 32
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1 - patch_size + 1) * (image_size - patch_size + 1 - patch_size + 1)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    theta = convolution_layer.initialize_two_conv_layers(hidden_size, visible_size, patch_size)

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    pattern = convolution_layer.generate_patch_pattern(patch_size, image_size - patch_size + 1, image_size - patch_size + 1)

    l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N, pattern)

    # l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_cost_two_conv_layers(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N, pattern)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_conv_and_lrf_layers():
    image_size = 32
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1 - patch_size + 1) ** 2 * (patch_size - small_patch + 1) ** 2

    theta = convolution_layer.initialize_conv_and_lrf_layers(hidden_size, visible_size, patch_size, small_patch, image_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    pattern = np.zeros((small_patch, small_patch, (image_size - patch_size + 1 - patch_size + 1) ** 2))
    r = np.sqrt(6) / np.sqrt(small_patch + small_patch + 1)
    for j in range((image_size - patch_size + 1 - patch_size + 1) ** 2):
        pattern[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r

    print "pattern.size: ", pattern.shape

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_and_lrf_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, convolution_layer.N, pattern)

    # l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_and_lrf_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, convolution_layer.N, pattern)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_conv_and_maxout():
    image_size = 32
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = conv_maxout.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1 - patch_size + 1) ** 2 * (patch_size - small_patch + 1) ** 2

    theta = conv_maxout.initialize_conv_and_lrf_layers(hidden_size, visible_size, patch_size, small_patch, image_size)

    pooled_size = (image_size - patch_size + 1)/conv_maxout.pool_size

    pattern = np.zeros((small_patch, small_patch, (image_size - patch_size + 1 - patch_size + 1) ** 2))
    r = np.sqrt(6) / np.sqrt(small_patch + small_patch + 1)
    for j in range((image_size - patch_size + 1 - patch_size + 1) ** 2):
        pattern[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r

    print "pattern.size: ", pattern.shape

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", conv_maxout.N

    l_cost, l_grad = conv_maxout.convolution_and_lrf_cost(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, conv_maxout.N, pattern)

    # l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: conv_maxout.convolution_and_lrf_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, conv_maxout.N, pattern)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_conv_and_maxout_log_regression():
    image_size = 32
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1 - patch_size + 1) ** 2 * (patch_size - small_patch + 1) ** 2

    theta = convolution_layer.initialize_conv_and_lrf_layers_log_regression(hidden_size, visible_size, patch_size, small_patch, image_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    pattern = np.zeros((small_patch, small_patch, (image_size - patch_size + 1 - patch_size + 1) ** 2))
    r = np.sqrt(6) / np.sqrt(small_patch + small_patch + 1)
    for j in range((image_size - patch_size + 1 - patch_size + 1) ** 2):
        pattern[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r

    print "pattern.size: ", pattern.shape

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_and_lrf_cost_log_regression(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, convolution_layer.N, pattern)

    # l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_and_lrf_cost_log_regression(x, visible_size, hidden_size, nju, lambda_, images, patch_size, small_patch, image_size, pooled_size, convolution_layer.N, pattern)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_two_conv_log_regression():
    image_size = 32
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1 - patch_size + 1) ** 2

    theta = convolution_layer.initialize_two_conv_layers_log_regression(hidden_size, visible_size, patch_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    pattern = np.zeros((small_patch, small_patch, (image_size - patch_size + 1 - patch_size + 1) ** 2))
    r = np.sqrt(6) / np.sqrt(small_patch + small_patch + 1)
    for j in range((image_size - patch_size + 1 - patch_size + 1) ** 2):
        pattern[:, :, j] = np.random.random((small_patch, small_patch)) * 2 * r - r

    print "pattern.size: ", pattern.shape

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers_log_regression(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N, pattern)

    # l_cost, l_grad = convolution_layer.convolution_cost_two_conv_layers(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_cost_two_conv_layers_log_regression(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N, pattern)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    # print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    # print "computed_grad: ", compute_grad
    # print "l_grad: ", l_grad[:20]


def check_grad_in_convolution_layer_with_pooling():
    image_size = 32
    patch_size = 9
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1) * (image_size - patch_size + 1)

    theta = convolution_layer.initialize_with_pooling(hidden_size, visible_size, patch_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_cost_with_pooling(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_cost_with_pooling(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    computed_grad = compute_grad(J, theta)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])

    print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad[:20]


def check_grad_in_convolution_layer_not_shared_weights():
    image_size = 32
    patch_size = 9
    num_labels = 10
    lambda_ = 0.1
    nju = 3e-3

    images = convolution_layer.prepare_data()

    visible_size = image_size * image_size
    hidden_size = (image_size - patch_size + 1) * (image_size - patch_size + 1)

    theta = convolution_layer.initialize_not_shared_weights(hidden_size, visible_size, patch_size)

    pooled_size = (image_size - patch_size + 1)/convolution_layer.pool_size

    # J = lambda x: deep_autoencoder.deep_autoencoder_cost(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, deep_autoencoder.N)
    # options_ = {'maxiter': 2, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print "N = ", convolution_layer.N

    l_cost, l_grad = convolution_layer.convolution_cost_not_shared_weights(theta, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    J = lambda x: convolution_layer.convolution_cost_not_shared_weights(x, visible_size, hidden_size, nju, lambda_, images, patch_size, image_size, pooled_size, convolution_layer.N)

    computed_grad = compute_grad(J, theta, l_grad)
    # options_ = {'maxiter': 50, 'disp': True}
    # result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
    # theta = result.x

    print np.linalg.norm(computed_grad - l_grad) / np.linalg.norm(computed_grad + l_grad)

    print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad


def check_gradient_in_sparse_autoencoder():
    patchsize = 32
    numpatches = 10000
    num_labels = 10

    print "Started reading file."
    data = scipy.io.loadmat('data/train_32x32.mat')
    print "Complete reading file."

    X_original = data['X']
    #y = data['y']

    row, col, rgb, m = X_original.shape
    X = np.zeros((m, row * col))

    # converting RGB into grayscale format
    for index in range(m):
        r, g, b = X_original[:, :, 0, index], \
            X_original[:, :, 1, index], X_original[:, :, 2, index]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        X[index, :] = gray.flatten()

    X /= 255.0

    #N = size-1 + 0.0
    images = X.T
    print "images.shape: ", images.shape

    train_data = images[:, :20]

    print 'Train_dat = ', train_data.shape

    visible_size = patchsize * patchsize
    #hidden_size = 324
    hidden_size = 196

    sparsity_param = 0.1
    lambda_ = 3e-3
    beta = 3

    #patches = sample_images()
    theta = sparse_autoencoder.initialize(hidden_size, visible_size)

    l_cost, l_grad = sparse_autoencoder.sparse_autoencoder_cost(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, train_data)

    J = lambda x: sparse_autoencoder.sparse_autoencoder_cost(x, visible_size, hidden_size, lambda_, sparsity_param, beta, train_data)

    computed_grad = compute_grad(J, theta)

    print np.linalg.norm(computed_grad - l_grad[:20]) / np.linalg.norm(computed_grad + l_grad[:20])
    # print "computed_grad: ", compute_grad
    print "l_grad: ", l_grad[:20]

# check_grad_in_deep_autoencoder()
# check_gradient_in_sparse_autoencoder()
# check_grad_in_convolution_layer()
# check_grad_in_convolution_layer_with_pooling()
# check_grad_in_convolution_layer_not_shared_weights()
# check_grad_in_two_convolution_layers()
# check_grad_in_conv_and_lrf_layers()
# check_grad_in_conv_and_maxout()
# check_grad_in_conv_and_maxout_log_regression()
# check_grad_in_two_conv_log_regression()
# check_grad_in_sparse_autoencoder()
# check_grad_in_sparse_smoothed_autoencoder()
# check_grad_in_k_deep_sparse_autoencoder()
