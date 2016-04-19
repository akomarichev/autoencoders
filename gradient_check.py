import numpy as np

# constants
EPSILON = 0.0001
S = 100


def compute_grad(J, theta, l_grad):
    for item in theta:
        grad_bp = []
        grad_n = []
        it = np.nditer(theta[item], flags=['multi_index'], op_flags=['readwrite'])
        print "Checking gradients for first ", S, " elements in ", item, '.'
        k = 0
        while not it.finished:
            index = it.multi_index

            grad_bp.append(l_grad[item][index])
            original_value = theta[item][index]

            theta[item][index] = original_value - EPSILON
            J_epsilon_minus = J(theta)[0]
            theta[item][index] = original_value + EPSILON
            J_epsilon_plus = J(theta)[0]
            theta[item][index] = original_value

            grad_n.append((J_epsilon_plus - J_epsilon_minus) / (2 * EPSILON))

            if k >= S:
                break
            it.iternext()
            k += 1

        print "Difference: ", np.linalg.norm(np.asarray(grad_bp) - np.asarray(grad_n)) / np.linalg.norm(np.asarray(grad_bp) + np.asarray(grad_n))
        print "Backpropagated:\n", np.asarray(grad_bp)
        print "Calculated numerically:\n", np.asarray(grad_n)
