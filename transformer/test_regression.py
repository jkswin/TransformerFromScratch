import unittest
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# relatives
from toy_data.dataset_wrapper import Datasets
from regression import LogRegressor, Regressor, NearestNeighbours, LOF

from sklearn.neighbors import LocalOutlierFactor


def create_test_dataset(n_samples=100, n_features=2, outliers_fraction=0.05, random_state=42):
        rng = np.random.RandomState(random_state)
        X = rng.randn(n_samples, n_features)
        n_outliers = int(n_samples * outliers_fraction)
        outliers_indices = rng.choice(range(n_samples), size=n_outliers, replace=False)
        X[outliers_indices] += 10 # Add outliers far away from the normal distribution
        return X, outliers_indices

class TestRegressor(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.X = np.array([
                    [0.2, 0.3],
                    [0.5, 0.1],
                    [0.8, 0.6],
                    [0.4, 0.9],
                    [0.7, 0.2],
                    [0.3, 0.5],
                    [0.1, 0.4],
                    [0.6, 0.8],
                    [0.9, 0.7],
                    [0.2, 0.1],
                    [0.4, 0.6],
                    [0.7, 0.3],
                    [0.5, 0.8],
                    [0.3, 0.2],
                    [0.1, 0.9],
                    [0.95, 0.95],
                    [1.8, 2.3],
                    [1.2, 1.4],
                ])


    @unittest.SkipTest
    def test_knn(self):
        nn = NearestNeighbours(X=self.X, k=3)
        neighbor_idxs, distances = nn.get_neighbours()
        print(neighbor_idxs, distances)

    @unittest.SkipTest
    def test_LOF(self):

        N_NEIGHBOURS = 5
        THRESHOLD = 1.5

        X, outliers_indices = create_test_dataset()
        
        lof = LOF(n_neighbours=N_NEIGHBOURS)
        outliers = lof.fit_predict(X, threshold=THRESHOLD)

        for idx in range(X.shape[0]):
            if idx in outliers[0]:
                colour = "magenta"
            else:
                colour = "green"

            plt.scatter(X[idx,0], X[idx,1], c=colour)

        plt.show()

        assert np.equal(outliers, outliers_indices)

    @unittest.SkipTest
    def test_linear(self):
        X_train, X_test, y_train, y_test, i_train, i_test, pokemon_df = Datasets.load_pokemon(
        train_size=0.9,
        upsample=True, # very few pokemon are legendary so I apply very basic smote upsampling
        )

        # HP is at col 0, and speed is at col 5 so lets do a simple regression
        X = np.expand_dims(pokemon_df["0"].to_numpy(), axis=1)
        y = pokemon_df["1"].to_numpy()

        lr  = Regressor(X=X, y=y, n_iters=20)
        initial_guess = lr.predict(X)
        losses = lr.fit(vis=True)
        print(f"\nInitial Loss: {losses[0]}\nFinal Loss: {losses[-1]}\n")
        plt.scatter(X,y,s=1)
        plt.plot(X, lr.predict(X), color="green")
        plt.plot(X, initial_guess, color="red")
        plt.xlabel("HP")
        plt.ylabel("Speed")
        plt.show()


    # currently failing as my implementation has a bug somewhere that I haven't tracked down 
    def test_linear_with_lof(self):
        """
        Comparing linear regression when using all data points vs fitting the line with outlier removal using Local Outlier Factor.
        """
        # hyperparameters
        N_NEIGHBOURS = 10
        THRESHOLD = 1.5
        
        #load in pokemon dataset
        X_train, X_test, y_train, y_test, i_train, i_test, pokemon_df = Datasets.load_pokemon(
        train_size=0.9,
        upsample=False,
        )
        
        # X = HP stats, y = Speed stats of all pokemon 
        X_full = np.expand_dims(pokemon_df["0"].to_numpy(), axis=1)
        y_full = pokemon_df["5"].to_numpy()

        # format the data so it can be passed to fit_predict in the write shape
        column_data = np.vstack((pokemon_df["0"].to_numpy(), y_full,)).T

        # use my implementation of Local Outlier Factor to get the indexes of the outliers
        lof = LOF(n_neighbours=N_NEIGHBOURS)
        outliers = lof.fit_predict(column_data, threshold=THRESHOLD, verbose=True)

        # compute using sklearn implementation for comparison
        lof_sk = LocalOutlierFactor(n_neighbors=N_NEIGHBOURS, algorithm="brute", metric="euclidean")
        outliers_sk = np.where(lof_sk.fit_predict(column_data) != 1)
        outliers = outliers_sk
        print(outliers)
        print(X_full[outliers])
        print(f"n_outliers: {outliers[0].shape[0]}")
        print(outliers_sk)
        print(pokemon_df.iloc[outliers][["name", "0", "5"]])

        # remove outliers to make a cropped version of X 
        X_cropped = np.delete(X_full.copy(), outliers, axis=0)
        y_cropped = np.delete(y_full.copy(), outliers, axis=0)

        # plot points, highlight the outliers by making them larger and pink
        for idx in range(X_full.shape[0]):
            if idx in outliers[0]:
                colour = "magenta"
                s=3
            else:
                colour = "blue"
                s=1

            plt.scatter(X_full[idx], y_full[idx], c=colour, s=s)

        # plot the regression line when fitting on the full data and on the data with the outliers removed
        for data in [(y_full, X_full, "blue"), (y_cropped, X_cropped, "green")]:
            y, X, colour = data
            lr  = Regressor(X=X, y=y, n_iters=100)
            losses = lr.fit(vis=False)
            # print the rmse loss to show that the overall loss is lower when the outliers are removed from the regression algorithm
            print(f"\nInitial Loss: {losses[0]}\nFinal Loss: {losses[-1]}\n")
            plt.plot(X, lr.predict(X), color=colour)
        
        plt.xlabel("HP")
        plt.ylabel("Speed")
        plt.title("Speed/HP\n(Blue=All Points, Green=No Outliers)")
        plt.show()

        


    @unittest.SkipTest
    def test_logistic(self):
        X_train, X_test, y_train, y_test, i_train, i_test, pokemon_df = Datasets.load_pokemon(
        train_size=0.9,
        upsample=True, # very few pokemon are legendary so I apply very basic smote upsampling
        )

        print("Labels:", set(y_train))
        unique, counts = np.unique(y_train, return_counts=True)
        print(counts)
    
        lr = LogRegressor(X=X_train, y=y_train, n_iters=1000)
        lr.fit()

        y_hat = lr.predict_class(X_test)
        f1 = f1_score(y_true=y_test, y_pred=y_hat)
        print(f"F1 : {f1}")

        cm = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_hat)
        plt.show()


if __name__ == "__main__":
    unittest.main()