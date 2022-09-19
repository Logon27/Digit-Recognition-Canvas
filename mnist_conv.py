from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, Softmax, Tanh
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import Network
from fileio import saveNetwork

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 1, 28, 28)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 60000)

# neural network
layers = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 800),
    Sigmoid(),
    Dense(800, 10),
    Softmax()
]

network = Network(layers, binary_cross_entropy, binary_cross_entropy_prime, x_train, y_train, epochs=25, learning_rate=0.1)
network.train()

# test
for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")

saveNetwork(network, "mnistNetwork.pkl")