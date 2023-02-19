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


class BaseLoss(object):

    def __init__(self) -> None:
        pass

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MSE(BaseLoss):

    def __init__(self) -> None:
        """ Initializes the MSE Loss Function """
        super().__init__()


    def __mse(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Mean Squared Error Loss Function """
        return np.mean(np.square(y_true - y_pred))


    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.__mse(y_true, y_pred)


    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size



class CrossEntropy(BaseLoss):

    def __init__(self) -> None:
        """ Initializes the Cross Entropy Loss Function """
        super().__init__()


    def __cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """ Cross Entropy Loss Function """
        return -np.sum(y_true * np.log(y_pred))


    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.__cross_entropy(y_true, y_pred)


    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_true / y_pred
