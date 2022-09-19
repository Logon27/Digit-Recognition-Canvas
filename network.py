#Imports For Image Processing To Randomize Inputs
from scipy.ndimage.interpolation import rotate, zoom, shift
import random
import numpy as np

class Network():

    def __init__(self, layers, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True, noise = True):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.noise = noise

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self):
        if self.verbose:
            print("Beginning training...")

        for e in range(self.epochs):
            error = 0
            for x, y in zip(self.x_train, self.y_train):
                #Input manipulation for randomness
                x = x.reshape(28, 28)
                x = self.randomRotateArray(x)
                x = self.randomShiftArray(x)
                x = self.randomClippedZoomArray(x)
                x = x.reshape(28 * 28, 1)

                # forward
                output = self.predict(x)

                # error
                error += self.loss(y, output)

                # backward
                grad = self.loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate)

            error /= len(self.x_train)
            if self.verbose:
                print(f"{e + 1}/{self.epochs}, error={error}")

    #Helper Functions To Randomize Training Inputs
    def randomRotateArray(self, x):
        x = rotate(x, angle=random.randint(-30, 30), reshape=False)
        return x

    def randomShiftArray(self, x):
        x = shift(x, shift=(random.randint(-3, 3),random.randint(-3, 3)))
        return x

    #Method from stackoverflow
    #https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    def randomClippedZoomArray(self, img, zoom_factor=random.uniform(0.75, 1.25), **kwargs):

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