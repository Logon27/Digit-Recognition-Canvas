from config import *
if enableCuda:
    import cupy as np
else:
    import numpy as np
from layer import Layer

class Flatten(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.flatten(input)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

    # Helper for debug printing
    def __str__(self):
        return self.__class__.__name__ + "(" + ")"