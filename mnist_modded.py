import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from activations import Sigmoid, Tanh
from losses import mse, mse_prime
from network import Network
from fileio import saveNetwork, loadNetwork

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# load MNIST copy for image display
(x_train_image, y_train_image), (x_test_image, y_test_image) = mnist.load_data()

# training data: 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)

# same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

num_samples = 1500

#layers
layers = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 35),
    Sigmoid(),
    Dense(35, 10),
    Sigmoid()
]

#network = loadNetwork("mnistNetwork.pkl")
network = Network(layers, mse, mse_prime, x_train, y_train, epochs=5, learning_rate=0.1)
network.train()

for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

#saveNetwork(network, "mnistNetwork.pkl")