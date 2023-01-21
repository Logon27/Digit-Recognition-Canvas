import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from nn import *
from nn.data_processing.image_utils import *

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

# Load MNIST copy for image display
(x_train_image, y_train_image), (x_test_image, y_test_image) = mnist.load_data()

x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

# Example layouts
# http://yann.lecun.com/exdb/mnist/

# Neural Network Layers
layers = [
    Convolutional((1, 28, 28), 5, 5),
    # Input Size = 28
    # Kernel Size = 5
    # Output Size = Input Size - Kernel Size + 1
    # 28 - 5 + 1 = 24
    Sigmoid(),
    Convolutional((5, 24, 24), 3, 5),
    Sigmoid(),
    Convolutional((5, 22, 22), 3, 5),
    Sigmoid(),
    # Reshape((2, 20, 20), (2 * 20 * 20, 1)),  # This is an alternative to Flatten
    Flatten((5, 20, 20)),
    Dense(5 * 20 * 20, 40),
    Sigmoid(),
    Dense(40, 10),
    Softmax()
]

#network = load_network("mnist-network.pkl")
# Setting the layer properties for every layer in the network.
conv_layer_properties = LayerProperties(learning_rate=0.01, optimizer=SGD(), weight_initializer=Normal(std=0.1), bias_initializer=Zero())
network = Network(
    layers,
    TrainingSet(x_train, y_train, x_test, y_test, np.argmax),
    loss=categorical_cross_entropy,
    loss_prime=categorical_cross_entropy_prime,
    epochs=5,
    batch_size=1,
    layer_properties=conv_layer_properties,
    data_augmentation=[lambda x: np.reshape(x, (28, 28)), lambda x: np.multiply(x, 255), random_rotate_array, random_shift_array, random_clipped_zoom_array, random_noise_array, lambda x: np.divide(x, 255), lambda x: np.reshape(x, (1, 28, 28))]
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