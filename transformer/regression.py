"""
Quick regression implementation.
"""


import numpy as np
from utils.activation_functions import sigmoid
from utils.loss_functions import rmse
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt


class Regressor():
    """
    Base class for fitting regression problems.
    """

    def __init__(self, X, y, n_iters=100, lr=0.001) -> None:
        self.X = X
        self.y = y

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        self.n_iters = n_iters
        self.W = np.zeros(X.shape[1])
        self.b = 0.0

        self.lr = lr
        

    def fit(self, vis=False):

        losses = []

        for _ in range(self.n_iters):
            y_hat = self.predict(self.X)
            dW = np.dot(self.X.T, (y_hat-self.y))
            db = np.sum(y_hat-self.y)/self.n_samples

            loss = rmse(self.y, y_hat)
            losses.append(loss)


            self.W -= self.lr * dW
            self.b -= self.lr * db

            
        # make line go down
        if vis:
            plt.plot(losses)
            plt.show()

        return losses


    def predict(self, X):
        return np.dot(X, self.W) + self.b

    
class LogRegressor(Regressor):
    """
    Modified Regressor for Binary Classification.
    """

    def __init__(self, X, y, n_iters=100, lr=0.001) -> None:
        super().__init__(X, y, n_iters, lr)

    def predict(self, X):
        return sigmoid(super().predict(X))
    
    def predict_class(self, X, threshold=0.5):
        return np.where(self.predict(X) > threshold, 1, 0)
    
    
if __name__ == "__main__":
    
    np.random.seed(12345)
    lin_problem = make_regression(n_samples=100, n_features=10, noise=10)
    lin = Regressor(lin_problem[0], lin_problem[1])
    lin.fit(vis=True)

    log_problem = make_classification(n_samples=100, n_features=10)
    log = LogRegressor(X=log_problem[0], y=log_problem[1])
    log.fit(vis=True)