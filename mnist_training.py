from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from network import Network
from dense import Dense
from activations import *
from losses import *
from fileio import *

def preprocess_data(x, y, limit):
    # Reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("longdouble") / 255
    # Encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# Print a numpy array for debugging
def printArray(npArray):
    print('\n'.join([''.join(['{:.2f} '.format(item) for item in row]) for row in npArray]))
    print("\n")

# Load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load MNIST copy for image display
(x_train_image, y_train_image), (x_test_image, y_test_image) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

# Convert to cupy arrays
if enableCuda:
    x_train, y_train = (np.asarray(x_train), np.asarray(y_train))
    x_test, y_test = (np.asarray(x_test), np.asarray(y_test))

# Neural Network Layers
layers = [
    Dense(28 * 28, 800),
    Sigmoid(),
    Dense(800, 10),
    Sigmoid(),
    Dense(10, 10),
    Softmax()
]

#network = loadNetwork("mnist-network.pkl")
network = Network(layers, mse, mse_prime, x_train, y_train, x_test, y_test, epochs=15, learning_rate=0.1)
network.train()
saveNetwork(network, "mnist-network.pkl")

# Visual Debug After Training
rows = 5
columns = 10
fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=False, sharey=True, figsize=(12, 8))
fig.canvas.manager.set_window_title('Network Predictions')
# "i" represents the test set starting index.
i = 0
for j in range(rows):
    for k in range(columns):
        output = network.predict(x_test[i])
        prediction = np.argmax(output)
        # Convert to a string to prevent an error with cupy
        prediction = str(prediction)
        axes[j][k].set_title(prediction)
        axes[j][k].imshow(x_test_image[i], cmap='gray')
        axes[j][k].get_xaxis().set_visible(False)
        axes[j][k].get_yaxis().set_visible(False)
        i += 1
plt.show()