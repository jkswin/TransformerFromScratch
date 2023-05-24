"""
Quick regression implementation.
"""


import numpy as np
from utils.activation_functions import sigmoid
from utils.loss_functions import rmse
from sklearn.datasets import make_regression, make_classification
import matplotlib.pyplot as plt
from utils.utils import timer


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
        

    def fit(self, vis=False, ignore_outlier=False):

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
    
    def _ignore_outliers(self):
        # calculate local outlier factor for data
        raise NotImplementedError()
        self.outliers = None
    
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
    

class NearestNeighbours:
    """
    Very basic brute force implementation of KNN.
    """
    def __init__(self, X, k=5):
        # to account for nearest neighbour being the data point itself
        self.k = k + 1
        self.X = X

    def _euclid(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _knn(self, x):
        # get the k nearest neighbours for a single data point
        distances = [self._euclid(x, data_point) for data_point in self.X]
        sorted_idxs = np.argsort(distances)
        kn_idxs = sorted_idxs[:self.k]
        kn_distances = np.take(np.array(distances), kn_idxs)
        return kn_idxs, kn_distances
    
    def get_neighbours(self):
        """_summary_

        :return: indexes, distances
        :rtype: _type_
        """
        idxs = []
        distances = []
        for x in self.X:
            kn_idxs, kn_distances = self._knn(x)
            idxs.append(kn_idxs)
            distances.append(kn_distances)

        return np.array(idxs), np.array(distances)

    

class LOF:
    """
    Local Outlier Factor implementation for outlier detection in Regression.
    https://dl.acm.org/doi/pdf/10.1145/335191.335388
    """
    def __init__(self, n_neighbours=20) -> None:
        """_summary_

        :param n_neighbours: _description_, defaults to 20
        :type n_neighbours: int, optional
        """
        self.n_neighbours = n_neighbours

    @timer
    def fit_predict(self, X, threshold=1.5, verbose=False):
        """_summary_

        :param X: _description_
        :type X: _type_
        :return: _description_
        :rtype: _type_
        """
        # get the distances of the kth neighbour. e.g. the furthest neighbour to consider
        distances_of_kth_neighbour, indexes_of_neighbours = self._k_distances(X)
        # 
        neighbouring_values = X[indexes_of_neighbours]
        rd = self._reachability_distances(distances_of_kth_neighbour, neighbouring_values)
        densities = self._local_reachability_density(rd, indexes_of_neighbours)
        factors = self._local_outlier_factor(densities, indexes_of_neighbours)
        if verbose:
            print(factors)
        outliers = self._label_outliers(factors, threshold=threshold)
        return np.where(outliers==1)

    def _k_distances(self, X):
        """_summary_

        :param X: _description_
        :type X: _type_
        :return: _description_
        :rtype: _type_
        """
        neighbors = NearestNeighbours(X, k=self.n_neighbours)
        indexes, distances = neighbors.get_neighbours()
        k_distances = distances[:,-1] # k distances is 1d array of length n_samples in X. It is the distance to the k'th data point
        return k_distances, indexes

    
    def _reachability_distances(self, k_distances, neighbours):
        """
         The maximum of the k-distance of the sample and the distance between the sample and its k-th nearest neighbor. 
         Used to determine how isolated or reachable a data point is within its neighborhood.
        """
        rds = np.zeros_like(k_distances)
        for idx, sample_neighbours in enumerate(neighbours):
            rds[idx] = np.max(np.maximum(sample_neighbours[1:], k_distances[idx])) 
        return rds 
    
    def _local_reachability_density(self, reachability_distances, neighbours):
        """_summary_

        :param reachability_distances: _description_
        :type reachability_distances: _type_
        :param neighbours: _description_
        :type neighbours: _type_
        :return: _description_
        :rtype: _type_
        """
        lrd = np.zeros_like(reachability_distances)
        for i, sample_neighbours in enumerate(neighbours):
            lrd[i] = 1.0 / (np.mean(reachability_distances[sample_neighbours[1:]]) + 1e-10)
        return lrd
    
    def _local_outlier_factor(self, lrd, neighbours):
        """_summary_

        :param lrd: _description_
        :type lrd: _type_
        :param neighbours: _description_
        :type neighbours: _type_
        """
        lof = np.zeros_like(lrd)
        for i, sample_neighbours in enumerate(neighbours):
            lof[i] = np.mean(lrd[sample_neighbours[1:]]) / lrd[i]
        return lof
    
    def _label_outliers(self, lof, threshold=1.5):
        """_summary_

        :param lof: _description_
        :type lof: _type_
        :param threshold: _description_, defaults to 1.5
        :type threshold: float, optional
        :return: _description_
        :rtype: _type_
        """
        return np.where(lof > threshold, 1, 0)


    
    


if __name__ == "__main__":
    
    np.random.seed(12345)
    lin_problem = make_regression(n_samples=100, n_features=10, noise=10)
    lin = Regressor(lin_problem[0], lin_problem[1])
    lin.fit(vis=True)

    log_problem = make_classification(n_samples=100, n_features=10)
    log = LogRegressor(X=log_problem[0], y=log_problem[1])
    log.fit(vis=True)