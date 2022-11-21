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
from losses import BaseLoss
from layers import BaseLayer


class Network(object):

    def __init__(self) -> None:
        """ Initializes the Network instance """
        self.layers: list[BaseLayer] = []
        self.loss = None


    def add(self, layer: BaseLayer) -> None:
        """ Adds a layer to the network """
        self.layers.append(layer)


    def compile(self, loss: BaseLoss):
        self.loss = loss


    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = None, learning_rate: float = 0.01, verbose = False) -> np.ndarray:
        """ Trains the network """
        errs = []
        for epoch in range(epochs):
            err = 0
            for x, _y in zip(X, y):
                # Forward Propagation
                output = x
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # Compute Error
                err += self.loss.forward(_y, output)

                # Backward Propagation
                error = self.loss.backward(_y, output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            if verbose:
                errs.append(err / len(X))
                print(f"Epoch {epoch + 1}/{epochs}\tError: {errs[-1]}")
        return errs


    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predicts the output of the network """
        y_pred = []
        for _x in X:
            output = _x
            for layer in self.layers:
                output = layer.forward_propagation(output)  # Forward propagation of the input
            y_pred.append(output)                           # Append the output to the result list

        return np.array(y_pred).reshape((len(X), -1))