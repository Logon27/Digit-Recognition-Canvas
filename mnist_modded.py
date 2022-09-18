import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from activations import Sigmoid, Softmax, Tanh
from losses import mse, mse_prime
from network import Network
from fileio import saveNetwork, loadNetwork

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 10000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

#784-2500-2000-1500-1000-500-10

#layers
layers = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 150),
    Sigmoid(),
    Dense(150, 35),
    Sigmoid(),
    Dense(35, 10),
    Softmax()
    # Dense(28 * 28, 2500),
    # Sigmoid(),
    # Dense(2500, 2000),
    # Sigmoid(),
    # Dense(2000, 1500),
    # Sigmoid(),
    # Dense(1500, 1000),
    # Sigmoid(),
    # Dense(1000, 500),
    # Sigmoid(),
    # Dense(500, 10),
    # Softmax()
]

#network = loadNetwork("mnistNetwork.pkl")
network = Network(layers, mse, mse_prime, x_train, y_train, epochs=10, learning_rate=0.1)
network.train()

for x, y in zip(x_test, y_test):
    output = network.predict(x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

saveNetwork(network, "mnistNetwork.pkl")