import unittest
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# relatives
from toy_data.dataset_wrapper import Datasets
from regression import LogRegressor, Regressor



class TestRegressor(unittest.TestCase):

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