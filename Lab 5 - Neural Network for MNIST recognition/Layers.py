from abc import abstractmethod, ABC
import numpy as np
from typing import Tuple

class Layer(ABC):
    def __init__(self):
        super().__init__()
        self.input_vector = None
        self.output_vector = None
        # TODO: Write abstract method __str__

    @abstractmethod
    def forward_propagation(self, input_vector):
        pass

    @abstractmethod
    def backward_propagation(self, output_gradient, learning_rate):
        pass

    @staticmethod
    def initiate_weights(input_size: int, output_size: int, uniform_range: Tuple[int, int]) -> np.array:
        """
        Generates randomized weights for connections in a specific layer.
        Each number is generated from uniform distribution.
        :param input_size: Number of neurons in the input of the layer.
        :param output_size: Number of neurons in the output of the layer.
        :param uniform_range: Range of uniform distribution
        """
        return np.random.uniform(uniform_range[0], uniform_range[1], (output_size, input_size))

    @staticmethod
    def initiate_biases(nn_input: int) -> np.array:
        """
        Returns a vector of biases, each initiated to zero.
        """
        return np.zeros((nn_input, 1))


class Hidden(Layer):  # hidden layer
    def __init__(self, input_size, output_size, activation_function, activation_prime):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self.initiate_weights(self.input_size, self.output_size, (-1, 1))
        self.biases = self.initiate_biases(self.output_size)
        self.activation_function = activation_function
        self.activation_prime = activation_prime

    def forward_propagation(self, input_vector):
        self.input_vector = input_vector

        y = self.biases + np.dot(self.weights, self.input_vector)
        return self.activation_function(y)

    def backward_propagation(self, previous_layer_output_gradient, learning_rate):
        """
        Updates weights: calculate derivative of an error with respect to the weights,
        using the chain rule, it is seen, that it equals to the derivative of an error,
        with respect to the previous layer output (which is given) times the derivative
        of previous layer output with respect the weights. After reduction,
        it is seen, that it simply equals to the derivative of an error, with respect to
        the previous layer output times transpose of an input vector.
        Update biases: calculate derivative of an error with respect to the biases,
        using the chain rule, it is seen, that it is simply derivative of an error
        with respect to the output vector from the previous layer.
        """
        dE_dW = np.dot(previous_layer_output_gradient, np.transpose(self.input_vector))  # Error derv with respect to weights

        self.weights -= learning_rate * dE_dW
        self.biases -= learning_rate * previous_layer_output_gradient

        dE_dX = np.dot(np.transpose(self.weights), previous_layer_output_gradient)

        # apply activation function
        next_output = np.multiply(dE_dX,
                                  self.activation_prime(self.input_vector))

        return next_output


class Input(Hidden):
    def __init__(self, output_size, activation_function, activation_prime):
        super().__init__(0, output_size, activation_function, activation_prime)


class Softmax(Layer):
    # TODO: improve softmax layer
    def __init__(self):
        super().__init__()

    def forward_propagation(self, input_vector):
        exp_val = np.exp(input_vector)
        self.output_vector = exp_val / np.sum(exp_val)
        return self.output_vector

    def backward_propagation(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output_vector)
        tmp = np.tile(self.output_vector, n)
        gradient = np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)
        return gradient


def sigmoid(x):
    try:
        value = 1 / (1 + np.exp(-x))
    except RuntimeWarning:
        print(x)
    else:
        return value


def sigmoid_prime(x):
    return np.exp(-x)/((np.exp(-x)+1)**2)
