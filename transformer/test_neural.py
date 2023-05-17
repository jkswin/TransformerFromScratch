"""
Test Cases for Neural Module. Non-assertive. 
"""


import unittest
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification

import matplotlib.pyplot as plt
import numpy as np

# relatives
from neural import *
from toy_data.dataset_wrapper import Datasets
from utils.loss_functions import rmse


class TestNeural(unittest.TestCase):

    @unittest.SkipTest
    def test_ffn(self):
        X_train, X_test, y_train, y_test, i_train, i_test, pokemon_df = Datasets.load_pokemon(
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
            activations= ["elu", "elu", "sigmoid"],
            loss = "log",
            seed=42,
            lr=0.1,
            )
        
        
        losses = net.train(X=X_train, y=y_train, epochs=20, batch_size=64, norm=True)

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

        misclass = pokemon_df.loc[i_test]
        misclass["pred"] = y_hat
        print(misclass[["name", "legend", "pred"]][misclass["legend"] != misclass["pred"]])

    def test_ffn_multinomial(self):
        
        X,y = make_classification(n_samples=5000, n_classes=5, random_state=42, n_features=20, n_informative=15)

        n_classes = np.unique(y).shape[0]
        dimension_of_x = X.shape[1]
        y = y.reshape(y.shape[0], 1)
        y_one_hot = np.zeros([y.shape[0], n_classes])
        for idx, row in enumerate(y):
            y_one_hot[idx][row[0]] = 1
        
    
        net = FFNeuralNet(
                layer_sizes= [dimension_of_x, 64, 64, n_classes],
                activations= ["elu", "elu", "softmax"],
                loss = "log",
                seed=42,
                lr=0.01,
                )
        
        losses = net.train(X=X, y=y_one_hot, epochs=100, batch_size=64, norm=True)
        plt.plot(range(len(losses)), losses)
        plt.xlabel("Epoch")
        plt.ylabel(net.loss.__name__.title() + " Loss")
        plt.show()

        y_hat = net.multinomial_classification(X)

        print(f"RMSE: {rmse(y_pred=y_hat, y_true=y)}")
        print("Raw Acc: ", np.sum(np.equal(y, y_hat))/y.shape[0])


    @unittest.SkipTest
    def test_forward_layer_forward(self):
        inputs = np.ones([2,4]) # two data points with 4 features each 
        expected_output = inputs * 5
        input_dim, output_dim = inputs.shape[1], inputs.shape[1] # 4,4
        layer = FFLayer(input_dim, output_dim, "linear", lr=0.1, seed=42)
        layer.W = np.zeros([input_dim, output_dim]) + 1
        layer.b = np.zeros([1, output_dim]) + 1

        # dot product = 4 
        # + bias of 1 
        # therefore array of 5s should be the output
        assert np.array_equal(layer.forward(inputs), expected_output)

    @unittest.SkipTest
    def test_rc_layer_forward(self):
        
        # create an instance of RCLayer
        rc_layer = RCLayer(input_dim=3, output_dim=2, activation="tanh", lr=0.1)

        # create some sample inputs
        inputs = np.array([
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
            [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8]]
        ])

        print(inputs.shape)

        # call the forward method to get the output
        output = rc_layer.forward(inputs)

        # print the output
        print(output)

    @unittest.SkipTest
    def test_rcnet(self):

        yugioh_names, one_hot_names = Datasets.load_yugioh()
        



if __name__ == "__main__":
    unittest.main()