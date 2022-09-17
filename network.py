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