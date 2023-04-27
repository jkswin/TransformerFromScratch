import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
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

        pokemon = pd.read_csv(
            NAME_PATH["pokemon"]
        )

        X, y = pokemon.drop(["legend", "name"], axis=1).to_numpy(), pokemon["legend"].to_numpy()

        if upsample:
            oversample = SMOTE()
            X,y = oversample.fit_resample(X,y)
            

        if not train_size:
            return X, y


        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, shuffle=True)  

        return X_train, X_test, y_train,  y_test
    


if __name__ == "__main__":
    data = Datasets.load_pokemon()
    for k, v in data.items():
        print(f"{k}: {v.shape}")