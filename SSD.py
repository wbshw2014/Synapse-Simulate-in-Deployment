import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

import numpy as np
from params import para_
from mydataset import dataset
from model import weights
from file import profile

class device:

    @staticmethod
    def inc(x,a,b,G_min,G_max):
        return x+a*np.exp(-b*(x-G_min)/(G_max-G_min))

    @staticmethod
    def dec(x,a,b,G_min,G_max):
        return x-a*np.exp(-b*(G_max-x)/(G_max-G_min))


def rand_G(weights, G_min, G_max, n_out, n_in):
    # np.random.seed()
    G = np.random.rand(n_out, n_in)
    G = (G - np.min(G)) / (np.max(G) - np.min(G))
    G = G * (G_max-G_min)+G_min
    # G /= np.sqrt(n_in)
    G[weights == 0] = G_min

    return G


def update_theta(a_inc, b_inc, a_dec, b_dec, G_min, G_max, theta, grad, G_plus, G_minus):

    G_plus[theta < grad] = device.inc(G_plus[theta < grad], a_inc, b_inc, G_min, G_max)
    G_minus[theta < grad] = device.dec(G_minus[theta < grad], a_dec, b_dec, G_min, G_max)

    G_plus[theta > grad] = device.dec(G_plus[theta > grad], a_dec, b_dec, G_min, G_max)
    G_minus[theta > grad] = device.inc(G_minus[theta > grad], a_inc, b_inc, G_min, G_max)

    G_plus[theta_ == theta_begin_] = G_min
    G_minus[theta_ == theta_begin_] = G_min

    G_plus[G_plus >= G_max] = G_max
    G_plus[G_plus <= G_min] = G_min

    G_minus[G_minus >= G_max] = G_max
    G_minus[G_minus <= G_min] = G_min

    G_plus[G_plus == G_minus] = G_min
    G_minus[G_plus == G_minus] = G_min

    return G_plus, G_minus


if __name__ == '__main__':

    np.random.seed(7)

    name_dataset = 'mnist'

    Epoch = 100
    input_sizes = 784
    num_labels = 10

    (train_images, train_labels), (test_images, test_labels) = dataset(name_dataset)

    model = Sequential([
        Flatten(input_shape=(input_sizes, num_labels)),
        Dense(num_labels, activation='softmax')
    ])

    optimizer = SGD(learning_rate=0.1)

    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy,
                  metrics=[SparseCategoricalAccuracy()])

    ########################################################################################################################

    Acc_train = []
    Cost_train = []

    G_plus_ = rand_G(weights, para_[-2], para_[-1], input_sizes + 1, num_labels)
    G_minus_ = rand_G(weights, para_[-2], para_[-1], input_sizes + 1, num_labels)

    G_plus_begin_ = G_plus_.copy()
    G_minus_begin_ = G_minus_.copy()
    theta_begin_ = (G_plus_ - G_minus_).copy()
    theta_best_ = G_plus_ - G_minus_
    theta_ = G_plus_ - G_minus_

    ########################################################################################################################

    for epoch in range(Epoch):

        G_plus_, G_minus_ = update_theta(*para_, theta_, weights, G_plus_, G_minus_)

        theta_ = G_plus_ - G_minus_
        W = [theta_[1::, :], theta_[0, :]]
        model.set_weights(W)

        J_train, A_train = model.evaluate(test_images, test_labels, verbose=0)
        Acc_train.append(A_train)
        Cost_train.append(J_train)

        accuracy = model.evaluate(test_images, test_labels, verbose=0)[1]
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")

        if A_train >= np.max(Acc_train):
            G_plus_best = G_plus_.copy()
            G_minus_best = G_minus_.copy()
            theta_best_ = G_plus_best - G_minus_best

        profile(theta_, model, name_dataset)


    print('Processing Done!')
