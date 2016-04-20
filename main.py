import numpy as np
import softmax
import sgd
import timeit

# constants

N = 72000
image_size = 32
patch_size = 4

sgd.run_sparse_autoencoder(N, image_size, patch_size, False)

# print timeit.timeit("run_sparse_autoencoder()", "from __main__ import run_sparse_autoencoder", number=1)

# softmax.run_softmax(N, image_size, patch_size)

# test()
