# imports 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
import pandas as pd


# jake's modules
from toy_data.dataset_wrapper import Datasets
from utils.activation_functions import relu, relu_prime, leaky_relu, leaky_relu_prime, \
                                sigmoid, sigmoid_prime, tanh, tanh_prime, \
                                linear, linear_prime, softmax, elu, elu_prime, softmax_prime, \
                                gelu, gelu_prime, relu_init, tanh_init

from utils.loss_functions import logloss, logloss_prime, mse, mse_prime, rmse
from utils.utils import *





    
class FFLayer:

    """
    A fully connected feed forward layer.

    Parameters
    ----------
    input_dim : int
        The number of input features.
    output_dim : int
        The number of output features.
    activation : str
        The activation function to use. Supported options are "tanh", "sigmoid",
        "relu", "leaky_relu", "linear", and "elu".
    lr : float
        The learning rate for the layer.
    seed : bool, optional
        Whether or not to seed the random number generator. Defaults to False.

    Attributes
    ----------
    activation_functions : dict
        A dictionary mapping supported activation function names to their
        corresponding activation and derivative functions.
    lr : float
        The learning rate for the layer.
    W : ndarray
        The weights for the layer, with shape (output_dim, input_dim).
    b : ndarray
        The biases for the layer, with shape (1, output_dim).
    act : function
        The activation function for the layer.
    act_prime : function
        The derivative of the activation function for the layer.

    Methods
    -------
    __init__(self, input_dim, output_dim, activation, lr, seed=False)
        Initializes the FFLayer object.
    __str__(self)
        Returns a string representation of the FFLayer object.
    forward(self, inputs)
        Computes the forward pass for the layer.
    backprop(self, dA)
        Computes the backward pass for the layer.

    """

    def __init__(self, input_dim: int, output_dim: int, activation: str, lr: float, seed=False) -> None:

        """
        Initializes the FFLayer object.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        output_dim : int
            The number of output features.
        activation : str
            The activation function to use. Supported options are "tanh", "sigmoid",
            "relu", "leaky_relu", "linear", and "elu".
        lr : float
            The learning rate for the layer.
        seed : bool or int, optional
            Whether or not to seed the random number generator. Defaults to False. 

        Returns
        -------
        None

        """

        self.activation_functions = {
        "tanh": [tanh, tanh_prime],
        "sigmoid": [sigmoid, sigmoid_prime],
        "relu": [relu, relu_prime], 
        "leaky_relu": [leaky_relu, leaky_relu_prime],
        "linear": [linear, linear_prime],
        "elu": [elu, elu_prime],
        "softmax": [softmax, softmax_prime],
        "gelu": [gelu, gelu_prime]
        }

        if seed:
            np.random.seed(seed)

        self.lr = lr
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(1, output_dim)
        self.act, self.act_prime = self.activation_functions.get(activation)
        self._weight_init(activation)

    def __str__(self) -> str:

        """
        Returns a string representation of the FFLayer object.

        Returns
        -------
        str
            A string representation of the FFLayer object.

        """

        return f"Inputs: {self.W.shape[1]}\nOutputs: {self.W.shape[0]}\nActivation: {self.act.__name__}\n"
    
    def _weight_init(self, activation:str):
        
        if activation.endswith("elu"):
            self.W = relu_init(self.W)    

        elif activation == "tanh":
            self.W = tanh_init(self.W)

        else:
            return


    def forward(self, inputs):

        """
        Computes the forward pass for the layer.

        Parameters
        ----------
        inputs : ndarray
            The inputs for the layer, with shape (batch_size, input_dim).

        Returns
        -------
        ndarray
            The outputs of the layer after the activation function is applied,
            with shape (batch_size, output_dim).

        """

        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.W.T) + self.b
        self.activations = self.act(self.outputs)

        return self.activations


    def backprop(self, dA):

        """
        Computes the backward pass for the layer.

        Parameters
        ----------
        dA : ndarray
            The upstream gradients for the layer, with shape (batch_size, output_dim).

        Returns
        -------
        ndarray
            The gradients with respect to the inputs for the layer, with shape
            (batch_size, input_dim).

        """

        dZ = self.act_prime(self.outputs) * dA #delta
        dW = 1/dZ.shape[0] * np.dot(dZ.T, self.inputs)
        db = 1/dZ.shape[0] * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ, self.W)

        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

        return dA_prev
    
    
    def layer_norm_forward(self, inputs, epsilon=1e-10):

        """
        Compute the forward pass for a fully connected feed forward layer with simplified layer normalization.
        
        Parameters
        ----------
        inputs : array_like
            The input to the layer, of shape (batch_size, input_dim).
        epsilon : float, optional
            A small value added to the denominator to prevent division by zero, defaults to 1e-10.
            
        Returns
        -------
        array_like
            The output activations of the layer after layer normalization, of shape (batch_size, output_dim).
            
        """

        x = inputs
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean(((x - mean) ** 2), axis=-1, keepdims=True)
        std = np.sqrt(var + epsilon)
        x = (x - mean) / std
        activations = self.forward(x)
    
        return activations


class FFNeuralNet:

    loss_functions = {

        "log":[logloss, logloss_prime],
        "mse":[mse, mse_prime],
    }

    def __init__(self, layer_sizes: List[int], activations: List[str], lr: int = 0.1, loss:str = "log", seed: int or None = None):

        if loss not in self.loss_functions.keys():
            raise ValueError(f"Loss function must be one of {list(self.loss_functions.keys())}")

        self.loss, self.loss_prime = self.loss_functions.get(loss)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = lr
        self.seed = seed
        self.layers = []
        self.costs = []
        self.epochs = None

        self.init_layers()
    

    def init_layers(self):

        assert len(self.layer_sizes) == len(self.activations) + 1, f"{len(self.layer_sizes)},{len((self.activations))}\nMake sure len(layer_sizes) is 1 larger than len(activations).\nThe first layer size is n_features."

        for i in range(len(self.layer_sizes)-1):

            self.layers += [FFLayer(self.layer_sizes[i], 
                                    self.layer_sizes[i+1], 
                                    self.activations[i],
                                    lr = self.lr,
                                    seed = self.seed,
                                    )
                                ]
            
            print(f"Layer {i}\n{self.layers[i]}")
            

    @timer
    def train(self, X: np.ndarray, y:np.ndarray, batch_size: int, epochs: int, norm=False):

        if norm:
            self.norm = norm
        else:
            self.norm = False
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]
            y = y[permutation]
            
            for batch_start in range(0, len(X), batch_size):
                batch_end = batch_start + batch_size
                batch_X = X[batch_start:batch_end]
                batch_y = y[batch_start:batch_end]
                
                # Forward pass
                a = batch_X

                for layer in self.layers:
                    if norm:
                        a = layer.layer_norm_forward(a)
                    else:
                        a = layer.forward(a)

                
                y_pred = a

                # Backward pass
                loss = self.loss(batch_y, y_pred)
                epoch_loss += loss

                dA = self.loss_prime(batch_y, y_pred)
                
                for layer in reversed(self.layers):
                    dA = layer.backprop(dA)
            if (epoch) % (epochs/5) == 0:
                print(f"Epoch {epoch}/{epochs}: loss={epoch_loss/len(X)}")
            losses.append(epoch_loss)
        
        return losses

    def predict(self, X): # where X is always a matrix
        A = X

        for layer in self.layers:
            if self.norm:
                A = layer.layer_norm_forward(A)
            else:
                A = layer.forward(A)

        return A
    

    def binary_classification(self, X, threshold=0.5):
        # where X is the output of a sigmoid output layer
        if self.activations[-1] != "sigmoid":
            raise ValueError("Output activation for binary classification must be sigmoid.")
        return np.where(self.predict(X) > threshold, 1, 0)
    

class RCLayer:
    """
    A Recurrent Layer.
    """
    def __init__(self, input_dim: int, output_dim: int, activation: str, lr: float, seed=False):

        self.activation_functions = {
            "tanh": [tanh, tanh_prime],
            "sigmoid": [sigmoid, sigmoid_prime],
            "relu": [relu, relu_prime], 
            "leaky_relu": [leaky_relu, leaky_relu_prime],
            "linear": [linear, linear_prime],
            "elu": [elu, elu_prime],
            "softmax": [softmax, softmax_prime],
            "gelu": [gelu, gelu_prime]
            }

        if seed:
            np.random.seed(seed)

        self.lr = lr
        self.Wxh = np.random.randn(output_dim, input_dim)

        self.b = np.random.randn(1, output_dim)
        self.act, self.act_prime = self.activation_functions.get(activation)

        



if __name__ == "__main__":

    X_train, X_test, y_train, y_test = Datasets.load_pokemon(
        train_size=0.9,
        upsample=True, # very few pokemon are legendary so I apply very basic smote upsampling
    )

    print("Labels:", set(y_train))
    unique, counts = np.unique(y_train, return_counts=True)
    print(counts)

    dimension_of_x = X_train.shape[1]
    n_samples = X_train.shape[0]
    print("No. of Samples:", n_samples)
    print("Number of Features:", dimension_of_x)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    net = FFNeuralNet(
        layer_sizes= [dimension_of_x, 64, 64, 1],
        activations= ["gelu", "gelu", "sigmoid"],
        loss = "mse",
        seed=42,
        lr=0.1,
        )
    
    
    losses = net.train(X=X_train, y=y_train, epochs=1000, batch_size=64, norm=True)
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel(net.loss.__name__.title() + " Loss")
    plt.show()

    y_hat = net.binary_classification(X_test)

    print(f"RMSE: {rmse(y_pred=y_hat, y_true=y_test)}")

    f1 = f1_score(y_true=y_test, y_pred=y_hat)
    print(f"F1 : {f1}")

    cm = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_hat)
    plt.show()
