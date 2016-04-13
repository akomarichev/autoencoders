import numpy as np
import cPickle
from itertools import chain
a = np.arange(6).reshape(2, 3)
indexes = np.argmax(a, axis=1)

# print a
# print indexes
# print indexes.shape

# print a[, indexes]


def f(a, N):
    return np.argsort(a, axis=0)[::-1, :][:N, :]

a = np.array([[4, 1, 0, 8, 8, 2], [3, 0, 2, 7, 7, 1], [9, 7, -5, 2, 3, 7]])
print a
print np.argsort(a, axis=0)
print np.argsort(a, axis=0)[::-1, :]
print np.argsort(a, axis=0)[::-1, :][:1, :]
mask = np.zeros(shape=(a.shape))
print mask
print f(a, 2)
k = f(a, 2)
# for i in range(a.shape[1]):
print "Split: ", np.hsplit(k, k.shape[1])
print "Split: ", np.hsplit(k, k.shape[1])
print range(a.shape[1])
# mask[np.hsplit(k, k.shape[1]), range(a.shape[1])] = 1
# mask[range(a.shape[0]), range(a.shape[1])] = 1
# print "Indeces: ", np.ix_(np.hsplit(k, k.shape[1]), range(a.shape[1]))
print np.arange(a.shape[1])
x = np.hsplit(k, k.shape[1])
y = range(a.shape[1])
print zip(x, y)
print zip(x, y)[0][0]
# print mask[l, :]
# print list(zip(np.hsplit(k, k.shape[1]), range(a.shape[1])))
# mask[k for k in zip(np.hsplit(k, k.shape[1]), range(a.shape[1]))] = 1

print [mask[i] for i in zip(np.hsplit(k, k.shape[1]), range(a.shape[1]))]

# for index in zip(np.hsplit(k, k.shape[1]), range(a.shape[1])):
#     mask[index] = 1

for i in range(a.shape[1]):
    mask[k[:, i], i] = 1

# mask[zip(np.hsplit(k, k.shape[1]), range(a.shape[1]))[:2]]

print f
print mask

# print a.shape[1]
# print mask
# print np.sum(mask == 0)
