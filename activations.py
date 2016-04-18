import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def rectifier(x):
    return np.maximum(0, x)


def rectifier_prime(x):
    return sign(rectifier(x))
