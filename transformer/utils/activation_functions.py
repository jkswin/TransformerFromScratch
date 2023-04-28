import numpy as np
from matplotlib import pyplot as plt
from scipy.special import erf

#TODO: refactor into classes

# softmax returns a probability distribution over classes so is used for multiclass classification

def relu(x):
    # helps prevent vanishing gradients
    return np.maximum(x, 0)

def relu_prime(x):
    # wrong for x=0 but this seems to be convention
    return np.where(x >= 0, 1.0, 0.0)

def leaky_relu(x, alpha=0.01): 
    # popular for tasks with sparse gradients. Helps prevent "dying ReLu" where neurons become inactive 
    x_alpha = alpha * x
    return np.where(x >= 0, x, x_alpha)

def leaky_relu_prime(x, alpha=0.01):
    return np.where(x>= 0, 1, alpha)

def sigmoid(x):
    # suffers from vanishing gradient problem so is unsuitable for deep networks
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    # produces zero centred output
    return np.tanh(x)

def tanh_prime(x):
    return 1 - tanh(x)**2

def linear(x):
    return x

def linear_prime(x):
    return 1

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax_prime(x):
    s = softmax(x)
    return np.diag(s) - np.outer(s, s)

# elu & prime implimented from https://arxiv.org/pdf/1511.07289.pdf
def elu(x, alpha=1.0):
    ex_x_alpha = alpha * (np.exp(x) - 1)
    return np.where(x >= 0, x, ex_x_alpha)

def elu_prime(x, alpha=1.0):
    x_alpha = elu(x) + alpha
    return np.where(x >= 0, 1, x_alpha)

def selu(x):
    raise NotImplementedError()

def swish(x):
    raise NotImplementedError()

def hard_tanh(x):
    raise NotImplementedError()

def gelu(x):
    return x * 0.5 * (1.0 + erf(x/np.sqrt(2.0)))

def gelu_prime(x):
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2/np.pi) * (x+0.044715 * np.power(x,3))))
    return 0.5 * cdf * (1.0 + np.sign(x))

def layer_norm(x, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x-mean)/(std+eps)


ALL_ACTIVATIONS = [relu, relu_prime, leaky_relu, leaky_relu_prime, 
    sigmoid, sigmoid_prime, tanh, tanh_prime, 
    linear, softmax, elu, elu_prime, gelu, gelu_prime]


# Playing with more complex weight initlization methods 
def relu_init(W: np.ndarray) -> np.ndarray:
    """
    Apply He initialization to the random weight matrix. 
    Used when the chosen non-linearity is ReLU.
    """
    return W * np.sqrt(2/(W.shape[0] - 1))


def tanh_init(W: np.ndarray) -> np.ndarray:
    """
    Apply Xavier initialization to the random weight matrix. 
    Used when the chosen non-linearity is TanH.
    """

    return W * np.sqrt(1/(W.shape[0] - 1))


if __name__ == "__main__":

    x = np.linspace(-10,10)

    plot_cols = 4
    plot_rows = 4
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(14,10))

    for ix in range(plot_rows):
        for iy in range(plot_cols):
            if ALL_ACTIVATIONS:
                fun = ALL_ACTIVATIONS.pop(0)
                axs[ix, iy].plot(x, fun(x), color="red", linewidth=3)
                axs[ix, iy].set_title(fun.__name__, size=8)
                axs[ix, iy].axhline(y=0, color = "k")
                axs[ix, iy].axvline(x=0, color = "k")
                
                

    plt.show()

    
    print(elu(np.array([[ 0, 1, -2, 3],
                         [-1, -15, 3, 4]]), alpha=0.9))
    