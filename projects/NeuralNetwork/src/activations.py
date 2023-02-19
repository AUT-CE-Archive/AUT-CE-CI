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


class BaseActivation(object):

    def __init__(self) -> None:
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError



class Sigmoid(BaseActivation):

    def __init__(self) -> None:
        """ Initializes the Sigmoid Activation Function """
        super().__init__()


    def __sigmoid(self, x: float):
        """ Sigmoid Activation Function """
        return 1 / (1 + np.exp(-x))


    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.__sigmoid(input)


    def backward(self, input: np.ndarray) -> np.ndarray:
        return self.__sigmoid(input) * (1 - self.__sigmoid(input))



class ReLU(BaseActivation):
    
    def __init__(self) -> None:
        """ Initializes the ReLU Activation Function """
        super().__init__()


    def __relu(self, x: float):
        """ ReLU Activation Function """
        return np.maximum(0, x)


    def forward(self, input: np.ndarray) -> np.ndarray:
         return self.__relu(input)


    def backward(self, input: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, 0)



class Tanh(BaseActivation):

    def __init__(self) -> None:
        """ Initializes the Tanh Activation Function """
        super().__init__()


    def __tanh(self, x: float):
        """ Tanh Activation Function """
        return np.tanh(x)


    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.__tanh(input)


    def backward(self, input: np.ndarray) -> np.ndarray:
        return 1 - (self.__tanh(input) ** 2)



class LeakyReLU(BaseActivation):

    def __init__(self) -> None:
        """ Initializes the Leaky ReLU Activation Function """
        super().__init__()


    def __leaky_relu(self, x: float):
        """ Leaky ReLU Activation Function """
        return max(0.01 * x, x)


    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.__leaky_relu(input)


    def backward(self, input: np.ndarray) -> np.ndarray:
        return np.where(input > 0, 1, 0.01)



class Softmax(BaseActivation):

    def __init__(self) -> None:
        super().__init__()


    def __softmax(self, x: np.ndarray):
        """ Softmax Activation Function """
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis = 0)


    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.__softmax(input)


    def backward(self, input: np.ndarray) -> np.ndarray:
        return self.__softmax(input) * (1 - self.__softmax(input))
