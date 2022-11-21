##########################################################
#
# Copyright (C) 2021-PRESENT: Keivan Ipchi Hagh
#
# Email:    keivanipchihagh@gmail.com
# GitHub:   https://github.com/keivanipchihagh
#
##########################################################

# Standard imports
import numpy as np

# Third-party imports
from activations import BaseActivation


class BaseLayer(object):
    """ Base Layer instance """

    def __init__(self, activation: BaseActivation = None) -> None:
        """ Initializes the Layer instance """
        self.activation = activation

    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        """ Forward propagation of the input for the layer """
        raise NotImplementedError

    def backward_propagation(self, output_error: np.ndarray, learning_rate: float):
        """ Backward propagation of the error for the layer and update parameters """
        raise NotImplementedError

    def _get_normal_matrix(self, width: int, height: int) -> np.ndarray:
        """ Returns the normalized weights of the layer """
        return np.random.rand(width, height) - 0.5

    def _get_zero_matrix(self, width: int, height: int) -> np.ndarray:
        """ Returns the zero matrix of the layer """
        return np.zeros((width, height))



class Dense(BaseLayer):
    """ Fully Connected (Dense) Layer """

    def __init__(self, input_size: int, output_size: int, activation: BaseActivation = None) -> None:
        """ Initializes the weights and biases for the layer """
        super().__init__(activation = activation)
        self.weights = self._get_normal_matrix(input_size, output_size)     # Normally distributed weights
        self.bias = self._get_zero_matrix(1, output_size)                   # All zeros bias


    def forward_propagation(self, input: np.ndarray) -> np.ndarray:
        """ Forward propagation of the batched input for the layer """
        self.input = input
        output = np.dot(self.input, self.weights) + self.bias   # Calculate forward propagation
        return self.activation.forward(output)                  # Apply activation function


    def backward_propagation(self, output_error: np.ndarray, learning_rate: float):
        """ Backward propagation of the error for the layer and update parameters """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        # Returns the backwared activated error
        return input_error