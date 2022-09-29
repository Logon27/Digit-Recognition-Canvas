from tracemalloc import start
from config import *
if enableCuda:
    print("Cuda Enabled.")
    import cupy as np
    from cupyx.scipy.ndimage import rotate, zoom, shift
else:
    print("Cuda Disabled.")
    import numpy as np
    from scipy.ndimage import rotate, zoom, shift
import random
import time
from PIL import Image as im

class Network():

    #The total training time in minutes.
    totalTrainingTime = 0

    def __init__(self, layers, loss, loss_prime, x_train, y_train, x_test, y_test, epochs = 1000, learning_rate = 0.01, verbose = True):
        self.layers = layers
        self.loss = loss
        self.loss_prime = loss_prime
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
    
    #For debug only
    # def saveImage(self, npArray, fileName):
    #     #Create an image from the array
    #     data = im.fromarray(npArray)
    #     data = data.convert("L")
        
    #     # saving the final output to file
    #     data.save(fileName)
    #     print("Saved Image... {}".format(fileName))

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self):
        if self.verbose:
            print("Beginning training...")
            startTime = time.time()

        for e in range(self.epochs):
            trainingError = 0
            for x, y in zip(self.x_train, self.y_train):
                #Input manipulation for randomness
                x = x.reshape(28, 28)
                x = np.multiply(x, 255)
                x = self.randomRotateArray(x)
                x = self.randomShiftArray(x)
                x = self.randomClippedZoomArray(x)
                x = self.randomNoiseArray(x)
                # self.saveImage(np.asnumpy(x), "output.png")
                # exit()
                x = x / 255
                x = x.reshape(28 * 28, 1)

                # forward
                output = self.predict(x)

                # error
                trainingError += self.loss(y, output)

                # backward
                grad = self.loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate)

            trainingError /= len(self.x_train)
            if self.verbose:
                ratioIncorrect = self.test()
                #Calculate estimated training time remaining for my sanity
                endTime = time.time()
                timeElapsedMins = (endTime - startTime) / 60
                timePerEpoch = timeElapsedMins / (e+1)
                epochsRemaining = self.epochs - (e+1)
                trainingTimeRemaining = timePerEpoch * epochsRemaining
                print("{}/{}, network training error = {:.4f}, test percentage incorrect = {:.2%}, training time remaining = {:.2f} minutes".format((e+1), self.epochs, trainingError, ratioIncorrect, trainingTimeRemaining))
        if self.verbose:
            endTime = time.time()
            timeElapsedMins = (endTime - startTime) / 60
            self.totalTrainingTime += timeElapsedMins
            print("Training Complete. Elapsed Time = {:.2f} seconds. Or {:.2f} minutes.".format(endTime - startTime, timeElapsedMins))

    #returns the ratio of incorrect responses in the training set
    def test(self):
        numCorrect = 0
        numIncorrect = 0
        for x, y in zip(self.x_test, self.y_test):
            output = self.predict(x)
            if np.argmax(output) == np.argmax(y):
                numCorrect+=1
            else:
                numIncorrect+=1
        return numIncorrect / (numCorrect + numIncorrect)

    #Helper Functions To Randomize Training Inputs
    def randomRotateArray(self, x):
        x = rotate(x, angle=random.randint(-20, 20), reshape=False)
        return x

    def randomShiftArray(self, x):
        x = shift(x, shift=(random.randint(-3, 3),random.randint(-3, 3)))
        return x
    
    # https://stackoverflow.com/questions/54633038/how-to-add-masking-noise-to-numpy-2-d-matrix-in-a-vectorized-manner
    def randomNoiseArray(self, x):
        frac = 0.005
        for i in range(5):
            randomInt = random.randint(50, 255)
            x[np.random.sample(size=x.shape) < frac] = randomInt
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