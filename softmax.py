import numpy as np
import os.path
import scipy.optimize
import scipy.sparse

import helper
import model

# constants
lambda_ = 0.01
options_ = {'maxiter': 500, 'disp': True}
num_labels = 10


def softmax_cost(theta, num_classes, input_size, lambda_, data, labels):

    m = data.shape[1]
    labels[labels == 10] = 0
    # print 'm: ', m
    theta = theta.reshape(num_classes, input_size)
    # print "theta.shape: ", theta.shape
    # print "data.hsape: ", data.shape
    theta_data = theta.dot(data)
    theta_data = theta_data - np.max(theta_data)
    # print "theta_data.shape: ", theta_data.shape
    prob_data = np.exp(theta_data) / np.sum(np.exp(theta_data), axis=0)
    # print "prob_data.shape: ", prob_data.shape
    # print "labels.shape: ", labels.shape
    # print "type(labels): ", type(labels)
    # print "np.unique(labels): ", np.unique(labels)
    indicator = scipy.sparse.csr_matrix((np.ones(m), (labels, np.array(range(m)))))
    # print "indicator.shape: ", indicator.shape
    indicator = np.array(indicator.todense())
    cost = (-1 / m) * np.sum(indicator * np.log(prob_data)) + (lambda_ / 2) * np.sum(theta * theta)
    # print "(indicator - prob_data).dot(data.transpose()).shape:", ((indicator - prob_data).dot(data.transpose())).shape
    grad = (-1 / m) * (indicator - prob_data).dot(data.transpose()) + lambda_ * theta

    return cost, grad.flatten()


def softmax_train(input_size, num_classes, lambda_, data, labels, options={'maxiter': 400, 'disp': True}):
    # Initialize theta randomly
    theta = 0.005 * np.random.randn(num_classes * input_size)

    J = lambda x: softmax_cost(x, num_classes, input_size, lambda_, data, labels)

    result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)

    print result
    # Return optimum theta, input size & num classes
    opt_theta = result.x

    return opt_theta, input_size, num_classes


def softmax_predict(model, data):
    opt_theta, input_size, num_classes = model
    opt_theta = opt_theta.reshape(num_classes, input_size)

    prod = opt_theta.dot(data)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    pred = pred.argmax(axis=0)

    return pred


def run_softmax(N, image_size, patch_size):

    # open training data
    print "Training data!"
    path = 'data/train_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_train.pickle"
    file_name_labels = "data/pickles/labels_train.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        helper.load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Training data were loaded and whitened!"

    images_train = helper.unpickle_data(file_name_data)[:, :, :N]
    labels_train = helper.unpickle_data(file_name_labels)[:N]

    # open test data
    print "Test data!"
    path = 'data/test_32x32.mat'
    file_name_data = "data/pickles/zca_whitened_test.pickle"
    file_name_labels = "data/pickles/labels_test.pickle"
    if not os.path.isfile(file_name_data) or not os.path.isfile(file_name_labels):
        helper.load_data_and_pickle(path, file_name_data, file_name_labels)
        print "Test data were loaded and whitened!"

    images_test = helper.unpickle_data(file_name_data)
    labels_test = helper.unpickle_data(file_name_labels)

    # open saved theta parameters
    theta = np.load('weights_learned/weights.out.npy')

    # get representations for training data
    file_name = "data/pickles/training_features.pickle"
    if not os.path.isfile(file_name):
        train_features = model.feed_forward(theta, images_train, patch_size, image_size, images_train.shape[2], 2)
        helper.pickle_data(file_name, train_features)
        print "Pickled training features!"
    else:
        train_features = helper.unpickle_data(file_name)
        print "Unpickled training features!"

    # get representations for test data
    file_name = "data/pickles/test_features.pickle"
    if not os.path.isfile(file_name):
        test_features = model.feed_forward(theta, images_test, patch_size, image_size, images_test.shape[2], 2)
        helper.pickle_data(file_name, test_features)
        print "Pickled testing features!"
    else:
        test_features = helper.unpickle_data(file_name)
        print "Unpickled testing features!"

    # print "Check gradients!"
    # lambda_ = 0.1
    # num_labels = 10
    # theta = 0.005 * np.random.randn(num_labels * image_size ** 2)
    # l_cost, l_grad = softmax_cost(theta, num_labels, image_size ** 2, lambda_, train_features, labels_train)
    # J = lambda x: softmax_cost(x, num_labels, image_size ** 2, lambda_, train_features, labels_train)
    # gradient_check.compute_grad(J, theta, l_grad)

    # run softmax function

    opt_theta, input_size, num_classes = softmax_train(image_size ** 2, num_labels, lambda_, train_features, labels_train, options_)
    predictions = softmax_predict((opt_theta, image_size ** 2, num_labels), test_features)
    labels_test[labels_test == 10] = 0
    print "Accuracy: {0:.2f}%".format(100 * np.sum(predictions == labels_test, dtype=np.float64) / labels_test.shape[0])
