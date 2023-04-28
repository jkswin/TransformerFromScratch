import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
import numpy as np 

NAME_PATH = {
    "pokemon":"transformer/toy_data/pokemon_data.csv"
}

@dataclass
class Datasets:

    def load_pokemon(train_size=0.8, upsample=False, random_state=42):
        """
        Load the Legendary Pokemon classification dataset. Covers Gen 1 to 8 of Pokemon. 
        Each Pokemon has 44 features. These are normalised base stats and one-hot encoded types. 

        :param train_size: _description_, defaults to 0.8
        :type train_size: float, optional
        :param upsample: _description_, defaults to False
        :type upsample: bool, optional
        :param random_state: _description_, defaults to 42
        :type random_state: int, optional
        :return: _description_
        :rtype: _type_
        """

        print("LOADING LEGENDARY POKEMON CLASSIFICATION...")

        pokemon_df = pd.read_csv(
            NAME_PATH["pokemon"]
        )

        X, y, i = pokemon_df.drop(["legend", "name"], axis=1).to_numpy(), pokemon_df["legend"].to_numpy(), pokemon_df.index.to_list()

        if upsample:
            oversample = SMOTE()
            len_y_prev = len(y) 
            X,y = oversample.fit_resample(X,y)
            additions = range(len_y_prev, len(y))
            i += additions

            new_X = X[len_y_prev:]
            new_y = y[len_y_prev:]

            new_df = pd.DataFrame(np.concatenate([new_X, np.zeros([new_y.shape[0], 1]), new_y.reshape([new_y.shape[0], 1])], axis=1))
            new_df.columns = pokemon_df.columns
            new_df["name"] = "fakemon"
            
            pokemon_df = pd.concat([pokemon_df, new_df], ignore_index=True)


        if not train_size:
            return X, y, i, pokemon_df


        X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, y, i, train_size=train_size, random_state=random_state, shuffle=True)  

        return X_train, X_test, y_train,  y_test, i_train, i_test, pokemon_df
    


if __name__ == "__main__":
    data = Datasets.load_pokemon(
        upsample=True
    )
    