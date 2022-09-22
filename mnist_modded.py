from config import *
if enableCuda:
    import cupy as np
    from cupyx.scipy.ndimage import rotate, zoom, shift
else:
    import numpy as np
    from scipy.ndimage.interpolation import rotate, zoom, shift
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from activations import LeakyRelu, Relu, Sigmoid, Softmax, Tanh
from losses import mse, mse_prime, binary_cross_entropy, binary_cross_entropy_prime
from network import Network
from fileio import saveNetwork, loadNetwork

#For image debugging
from PIL import Image as im
import random

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("longdouble") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def printArray(npArray):
    print('\n'.join([''.join(['{:.2f} '.format(item) for item in row]) for row in npArray]))
    print("\n")

# def saveImage(npArray, fileName):
#     #Create an image from the array
#     data = im.fromarray(npArray)
#     data = data.convert("L")
      
#     # saving the final output to file
#     data.save(fileName)
#     print("Saved Image... {}".format(fileName))

def saveImage(fileName, npArray):
    #Create an image from the array
    data = im.fromarray(npArray)
      
    # saving the final output to file
    data.save(fileName)
    print("Saved Image... {}".format(fileName))

def randomRotateArray(x):
    x = rotate(x, angle=random.randint(-30, 30), reshape=False)
    return x

def randomShiftArray(x):
    x = shift(x, shift=(random.randint(-3, 3),random.randint(-3, 3)))
    return x

#Method from stackoverflow
#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
def randomClippedZoomArray(img, zoom_factor=random.uniform(0.75, 1.25), **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

def addNoise(numpyArray):
    frac = 0.03
    randomInt = random.randint(50, 255)
    numpyArray[np.random.sample(size=numpyArray.shape) < frac] = randomInt
    return numpyArray

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# load MNIST copy for image display
(x_train_image, y_train_image), (x_test_image, y_test_image) = mnist.load_data()

# saveImage("before.png" , x_train[0])
#noise = np.random.normal(0,1,size=(28,28))
#x = randomClippedZoomArray(x_train[0])
#x = randomRotateArray(x)
#x = randomShiftArray(x)
# print(x_train[0].shape)
#printArray(x)
# x = addNoise(x_train[0])
# saveImage("after.png" , x)

x_train, y_train = preprocess_data(x_train, y_train, 2000)
x_test, y_test = preprocess_data(x_test, y_test, 2000)

#convert to cupy arrays
if enableCuda:
    x_train, y_train = (np.asarray(x_train), np.asarray(y_train))
    x_test, y_test = (np.asarray(x_test), np.asarray(y_test))

#784-2500-2000-1500-1000-500-10

# Dense(28 * 28, 70),
# Sigmoid(),
# Dense(70, 35),
# Sigmoid(),
# Dense(35, 10),
# Softmax()

#layers
layers = [
    # Dense(28 * 28, 70),
    # Sigmoid(),
    # Dense(70, 35),
    # Sigmoid(),
    # Dense(35, 10),
    # Sigmoid(),
    # Dense(10, 10),
    # Softmax()
    Dense(28 * 28, 800),
    LeakyRelu(),
    Dense(800, 10),
    LeakyRelu(),
    Dense(10, 10),
    Softmax()
]

#network = loadNetwork("mnistNetwork.pkl")
network = Network(layers, mse, mse_prime, x_train, y_train, x_test, y_test, epochs=5, learning_rate=0.1)
network.train()

# print("Running Against Test Dataset...")
# numCorrect = 0
# numIncorrect = 0
# for x, y in zip(x_test, y_test):
#     output = network.predict(x)
#     if np.argmax(output) == np.argmax(y):
#         numCorrect+=1
#     else:
#         numIncorrect+=1
#     #print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
# print("Test Dataset Accuracy: {}".format(numCorrect / (numCorrect + numIncorrect)))

saveNetwork(network, "mnistNetwork.pkl")

#Visual Debug
fig, axes = plt.subplots(ncols=20, sharex=False, sharey=True, figsize=(20, 4))
for i in range(20):
    output = network.predict(x_test[i])
    prediction = np.argmax(output)
    #Convert to a string to prevent an error with cupy
    prediction = str(prediction)
    
    axes[i].set_title(prediction)
    axes[i].imshow(x_test_image[i], cmap='gray')
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
plt.show()