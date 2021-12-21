from Layers import (Dense, Layer, Softmax,
                    sigmoid, sigmoid_prime)

import numpy as np

import file_management as fm


class NeuralNet:
    def __init__(self, learning_rate, epochs, loss_function, loss_prime):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self._loss_function = loss_function
        self._loss_prime = loss_prime

    def add(self, layer: Layer):
        self.layers.append(layer)

    def predict(self, input_vector):
        output_vector = input_vector
        for layer in self.layers:
            output_vector = layer.forward_propagation(output_vector)
        return output_vector

    def train(self, x_train, y_train, verbose=True):

        error = 0
        for epoch in range(1, self.epochs+1):
            classified_correctly_counter = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)

                # calculate error
                error += self._loss_function(y, output)

                classified_correctly_counter += int(np.argmax(output) == np.argmax(y))

                # calculate loss function
                grad = self._loss_prime(y, output)

                # propagate backward
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, self.learning_rate)

            error /= len(x_train)

            if verbose:
                print(f"Epoch {epoch}, classified correctly: {classified_correctly_counter}")
        print('Total error', error)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fm.load_data()
    x_train, y_train = fm.preprocess_data(x_train, y_train)
    x_test, y_test = fm.preprocess_data(x_test, y_test)

    nn = NeuralNet(learning_rate=0.02, epochs=5,
                   loss_function=mse, loss_prime=mse_prime)

    n_neurons = x_train.shape[1]

    nn.add(Dense(n_neurons, 20, sigmoid, sigmoid_prime))
    nn.add(Dense(20, 10, sigmoid, sigmoid_prime))
    nn.add(Softmax())

    nn.train(x_train, y_train, True)
