import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

from dense import Dense
from activations import Sigmoid, Softmax, Tanh
from losses import mse, mse_prime
from network import Network
from fileio import saveNetwork, loadNetwork
from scipy.ndimage.interpolation import rotate, zoom, shift

#For image debugging
from PIL import Image as im
import random

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

def printArray(npArray):
    print('\n'.join([''.join(['{:.2f} '.format(item) for item in row]) for row in npArray]))
    print("\n")

def saveImage(fileName, npArray):
    #Create an image from the array
    data = im.fromarray(npArray)
      
    # saving the final output to file
    data.save(fileName)

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

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#saveImage("before.png" , x_train[0])
#noise = np.random.normal(0,1,size=(28,28))
#x = randomClippedZoomArray(x_train[0])
#x = randomRotateArray(x)
#x = randomShiftArray(x)
#print(x.shape)
#printArray(x)
#saveImage("after.png" , x)

x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

#784-2500-2000-1500-1000-500-10

#layers
layers = [
    Dense(28 * 28, 150),
    Sigmoid(),
    Dense(150, 70),
    Sigmoid(),
    Dense(70, 35),
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