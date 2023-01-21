import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from nn import *
from nn.data_processing.image_utils import *
from copy import deepcopy

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

# Neural Network Layers
layers = [
    Dense(28 * 28, 400),
    Sigmoid(),
    Dense(400, 10),
    Sigmoid(),
    Dense(10, 10),
    Softmax()
]

#network = load_network("mnist-network.pkl")
network = Network(
    layers,
    TrainingSet(x_train, y_train, x_test, y_test, np.argmax),
    loss=mean_squared_error,
    loss_prime=mean_squared_error_prime,
    epochs=10,
    batch_size=1,
    layer_properties=LayerProperties(learning_rate=0.05, optimizer=SGD()),
    # The data augmentation de-normalizes the training sample then applies
    # shifting, rotation, translation, and noise during each iteration of training.
    # This is due to the mnist dataset being centered and normalized.
    data_augmentation=[lambda x: np.reshape(x, (28, 28)), lambda x: np.multiply(x, 255), random_rotate_array, random_shift_array, random_clipped_zoom_array, random_noise_array, lambda x: np.divide(x, 255), lambda x: np.reshape(x, (28 * 28, 1))]
)
network.train()
save_network(network, "mnist-network.pkl")

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