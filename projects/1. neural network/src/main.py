# Standard imports
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

# Third-party imports
from losses import MSE
from layers import Dense
from network import Network
from activations import Softmax, ReLU, Tanh, Sigmoid, LeakyReLU


def show_image(img: np.ndarray) -> None:
    """ Plots an image on grayscale """
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ Calculates the accuracy of the predictions """
    return sum(1 for x, y in zip(y_true, y_pred) if x == y) / len(y_true)


def subsample(x: np.ndarray, y: np.ndarray, n: int) -> tuple:
    """ Returns a subsample of the dataset """
    return x[:n], y[:n]



def reshape_and_normalize_data(data: np.ndarray) -> np.ndarray:
    """ Reshapes and normalizes the data """
    data = data.reshape(data.shape[0], 1, 28 * 28)
    return data.astype('float32') / 255

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
x_train = reshape_and_normalize_data(x_train)
x_test = reshape_and_normalize_data(x_test)



network = Network()                                         # Initialize network
network.add(Dense(28 * 28, 16, activation = ReLU()))        # Hidden layer #1
network.add(Dense(16, 16, activation = ReLU()))             # Hidden layer #2
network.add(Dense(16, 10, activation = Tanh()))             # Output layer
network.compile(MSE())                                      # Compile network



x, y_true = subsample(x_train, y_train, 100)     # Subsample the data

y_pred = network.predict(x)                 # Make predictions
y_pred = [np.argmax(_) for _ in y_pred]     # Get the labels of the predictions

# Calculate the accuracy
accuracy = accuracy(y_true, y_pred)
print(f'Accuracy: {round(accuracy * 100, 2)}%')