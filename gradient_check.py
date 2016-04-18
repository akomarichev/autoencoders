import numpy as np


def compute_grad(J, theta, l_grad):
    EPSILON = 0.0001

    grad = np.zeros(theta.shape, dtype=np.float64)

    print "grad.shape: ", grad.shape
    print "theta.shape: ", theta.shape

    dim2 = ((image_size - patch_size + 1) - patch_size + 1) * ((image_size - patch_size + 1) - patch_size + 1)
    # start = 2 ** 4 * 16 ** 2 + 4 ** 4 * 8 ** 2
    # start = 0
    start = 4 ** 4 * 8 ** 2 + 8 ** 4 * 4 ** 2 + 32 ** 4 + 2 * 32 ** 2
    # start = 8 ** 4 * 4 ** 2
    # start = patch_size * patch_size  # + dim2 * small_patch ** 2

    # theta.shape[0]
    # for i in range(theta.shape[0]):
    for i in range(start, start + 101):
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
