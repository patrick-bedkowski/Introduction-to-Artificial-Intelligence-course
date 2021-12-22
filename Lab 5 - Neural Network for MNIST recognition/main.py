import cProfile

from Layers import (Layer,
                    Dense,
                    Activation,
                    Softmax,
                    sigmoid,
                    sigmoid_prime,
                    relu,
                    relu_prime)

import numpy as np

import file_management as fm

np.seterr(all='warn')

class NeuralNet:
    def __init__(self, learning_rate, epochs, loss_function, loss_prime):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self._loss_function = loss_function
        self._loss_prime = loss_prime

    def add(self, layer: Layer):
        self.layers.append(layer)

    def propagate_through_layers(self, input_vector):
        output_vector = input_vector  # copy.copy(input_vector)
        for layer in self.layers:
            output_vector = layer.forward_propagation(output_vector)
        return output_vector

    def initiate_weights_of_input_layer(self, n_neurons):
        input_layer = self.layers[0]  # get first layer
        input_layer.input_size = n_neurons
        input_layer.weights = input_layer.initiate_weights(input_layer.input_size, input_layer.output_size, (-1, 1))

    def train(self, x_train, y_train, verbose=True):
        error = 0
        for epoch in range(1, self.epochs+1):
            classified_correctly_counter = 0
            h = 0
            for x, y in zip(x_train, y_train):
                output = self.propagate_through_layers(x)

                # calculate error
                error = self._loss_function(y, output)

                classified_correctly_counter += int(np.argmax(output) == np.argmax(y))

                # calculate loss function
                grad = self._loss_prime(y, output)

                # propagate backward
                for layer in reversed(self.layers):
                    grad = layer.backward_propagation(grad, self.learning_rate)
                h += 1

            error /= len(x_train)

            if verbose:
                cc_rate = round(classified_correctly_counter*100/len(y_train), ndigits=4)
                print(f"Epoch {epoch}, classified correctly: {cc_rate}, error {error}")
        # print('Total error', error)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def main():
    (x_train, y_train), (x_test, y_test) = fm.load_data()
    x_train, y_train = fm.preprocess_data(x_train, y_train)
    x_test, y_test = fm.preprocess_data(x_test, y_test)

    nn = NeuralNet(learning_rate=0.1, epochs=20,
                   loss_function=mse, loss_prime=mse_prime)

    nn.add(Dense(784, 40))
    nn.add(Activation(sigmoid, sigmoid_prime))
    nn.add(Dense(40, 10))
    nn.add(Activation(sigmoid, sigmoid_prime))
    nn.add(Softmax())

    nn.train(x_train, y_train, True)

    print('Training...')
    classified_correctly_counter = 0
    for x, y in zip(x_test, y_test):
        output = nn.propagate_through_layers(x)
        classified_correctly_counter += int(np.argmax(output) == np.argmax(y))
    cc_rate = round(classified_correctly_counter * 100 / len(y_test), ndigits=4)
    print(f'NN classified correctly: {cc_rate}% samples')


if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run('main()', 'output.dat')
    #
    # import pstats
    # from pstats import SortKey
    #
    # with open('output_time.txt', "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("time").print_stats()
    #
    # with open('output_calls.txt', "w") as f:
    #     p = pstats.Stats("output.dat", stream=f)
    #     p.sort_stats("calls").print_stats()

