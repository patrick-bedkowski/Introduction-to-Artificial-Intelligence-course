import pandas as pd
import time
from Layers import (Layer,
                    Dense,
                    Activation,
                    Softmax,
                    sigmoid,
                    sigmoid_prime)

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

    def propagate_through_layers(self, input_vector):
        output_vector = input_vector
        for layer in self.layers:
            output_vector = layer.forward_propagation(output_vector)
        return output_vector

    def initiate_weights_of_input_layer(self, n_neurons):
        input_layer = self.layers[0]  # get first layer
        input_layer.input_size = n_neurons
        input_layer.weights = input_layer.initiate_weights(input_layer.input_size, input_layer.output_size, (-1, 1))

    def fit(self, x_train, y_train, verbose=True):
        error = 0
        for epoch in range(1, self.epochs+1):
            classified_correctly_counter = 0
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

            error /= len(x_train)

            if verbose:
                cc_rate = round(classified_correctly_counter*100/len(y_train), ndigits=4)
                print(f"Epoch {epoch}, classified correctly: {cc_rate}, error {error}")


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def main():
    (x_train, y_train), (x_test, y_test) = fm.load_data()
    x_train, y_train = fm.preprocess_data(x_train, y_train)
    x_test, y_test = fm.preprocess_data(x_test, y_test)

    two_hidden_layers = [False, True]
    input_neurons_if_one_hidden_layer = [16, 25, 40]
    input_neurons_if_two_hidden_layers = [(30, 15), (15, 15), (15, 30)]
    epochs = 10
    learning_rate = 0.1

    results_df = pd.DataFrame(columns=['Epoch', 'Learning rate', 'Hidden layers',
                                       'N of input neurons in layer 1',
                                       'N of input neurons in layer 2',
                                       'Classified correctly', "Execution time"])

    # classification_results = pd.DataFrame(columns=['Predicted', 'Real'])

    for two_hid in two_hidden_layers:
        for two_layers, one_layers in zip(input_neurons_if_two_hidden_layers, input_neurons_if_one_hidden_layer):
            if two_hid:
                neurons = two_layers
            else:
                neurons = one_layers

            nn = NeuralNet(learning_rate=learning_rate, epochs=epochs,
                           loss_function=mse, loss_prime=mse_prime)

            if two_hid:
                nn.add(Dense(784, neurons[0]))
                nn.add(Activation(sigmoid, sigmoid_prime))
                nn.add(Dense(neurons[0], neurons[1]))
                nn.add(Activation(sigmoid, sigmoid_prime))
                nn.add(Dense(neurons[1], 10))
                nn.add(Activation(sigmoid, sigmoid_prime))
                nn.add(Softmax())
            else:
                nn.add(Dense(784, neurons))
                nn.add(Activation(sigmoid, sigmoid_prime))
                nn.add(Dense(neurons, 10))
                nn.add(Activation(sigmoid, sigmoid_prime))
                nn.add(Softmax())

            start_time = time.time()

            nn.fit(x_train, y_train, False)

            # TESTING
            classified_correctly_counter = 0
            for x, y in zip(x_test, y_test):
                output = nn.propagate_through_layers(x)

                predicted_number = np.argmax(output)
                real_number = np.argmax(y)
                # classification_result = {'Predicted': predicted_number,
                #                          'Real': real_number}

                # classification_results = classification_results.append(classification_result, ignore_index=True)

                classified_correctly_counter += int(np.argmax(output) == np.argmax(y))

            cc_rate = round(classified_correctly_counter * 100 / len(y_test), ndigits=4)

            exec_time = time.time() - start_time

            n_neurons_layer_1 = neurons[0] if two_hid else neurons
            n_neurons_layer_2 = neurons[2] if two_hid else neurons

            data_to_save = {'Epoch': epochs,
                            'Learning rate': learning_rate,
                            'Hidden layers': '2' if two_hid else '1',
                            'N of input neurons in layer 1': n_neurons_layer_1,
                            'N of input neurons in layer 2': n_neurons_layer_2,
                            'Classified correctly': cc_rate,
                            'Execution time': exec_time}

            results_df = results_df.append(data_to_save, ignore_index=True)

    results_df.to_csv('data_1.csv', index=False)
    # classification_results.value_count()
    # classification_results.to_csv('c_m_1.csv', index=False)
    results_df.to_latex(buf='table_1.txt', decimal=',')


if __name__ == '__main__':
    main()
