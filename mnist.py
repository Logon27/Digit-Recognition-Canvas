from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from activations import Sigmoid, Tanh
from losses import mse, mse_prime

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

# neural network
network = [
    Dense(28 * 28, 70),
    Sigmoid(),
    Dense(70, 35),
    Sigmoid(),
    Dense(35, 10),
    Sigmoid()
]

epochs = 5
learning_rate = 0.3
#not doing anything right now
num_samples = 1500

# train
for e in range(epochs):
    error = 0
    # train on 1000 samples, since we're not training on GPU...
    #for x, y in zip(x_train[:num_samples], y_train[:num_samples]):
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y, output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    #error /= num_samples
    error /= x_train.shape[0]
    print('%d/%d, error=%f' % (e + 1, epochs, error))

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for x, y in zip(x_test[:20], y_test[:20]):
    output = x
    for layer in network:
        output = layer.forward(output)
    prediction = np.argmax(output)
    correct_label = np.argmax(y)
    #print("Output: {}".format(output))
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
    # append correct or incorrect to list
    if (prediction == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
    
# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


fig, axes = plt.subplots(ncols=20, sharex=False, sharey=True, figsize=(20, 4))
for i in range(20):
    output = x_test[i]
    for layer in network:
        output = layer.forward(output)
    prediction = np.argmax(output)
    
    axes[i].set_title(prediction)
    axes[i].imshow(x_test_image[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()