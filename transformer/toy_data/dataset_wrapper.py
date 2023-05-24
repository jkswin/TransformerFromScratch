import os
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
import numpy as np 
import requests
import json

NAME_PATH = {
    "pokemon":"transformer/toy_data/pokemon_data.csv",
    "pokemon_nlp": "transformer/toy_data/pokemon_bios.json",
    "pokemon_legendary": "transformer/toy_data/pokemon_legendary.csv",
    "yugioh": "transformer/toy_data/yugioh_db.json",
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
    
    def load_pokemon_nlp() -> dict:
        """
        Load a list of 583 pokemon records with `name`, `desc` and `type`.
        Intended data use is to predict typing from natural language description. 

        :return: _description_
        :rtype: dict
        """

        print("LOADING POKEMON DESCRIPTIONS...")

        data_path = NAME_PATH["pokemon_nlp"]

        with open(data_path, "r") as f:
            data = json.load(f)

        pokemon_df = pd.read_csv(
            NAME_PATH["pokemon_legendary"]
        )
        
        records = []
        n_tokens = 0
        # replace pokemon names with a special token
        for k,v in data.items():
            if k in pokemon_df["Name"].to_list():
                for pok in list(data.keys()):
                    desc = v.replace(pok,"<POK>")
                desc = desc.strip()
                record = {"name":k, "desc":desc, "type":list(pokemon_df[pokemon_df["Name"] == k][["Type 1", "Type 2"]].values[0])}
                records.append(record)
                n_tokens += len(desc.split())

        print(f"n_pokemon: {len(records)}\nn_tokens: {n_tokens}")


        return records

    

    def load_yugioh(update=False) -> "tuple[list[str], list[np.ndarray]]":
        """
        Load a list of all current YugiOh Card names. 
        :return: _description_
        :rtype: _type_
        """

        api_endpoint = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
        data_path = NAME_PATH["yugioh"]

        if os.path.exists(data_path) and not update:
            with open(data_path, "r") as f:
                data = json.load(f)

        else:
            print(f"Downloading from: {api_endpoint}")
            data = json.loads(requests.get(api_endpoint).content)
            with open(data_path, "w") as f:
                json.dump(data, f)
        
        card_names = [card["name"] for card in data["data"]]

        # quick one hot encoding
        chars = {char:idx for idx, char in enumerate(["<S>", "<E>"] + sorted( list(set("".join(card_names).lower()))))}
        
        one_hot = []
        
        for card in card_names:
            card= ["<S>"] + [c for c in card.lower()] + ["<E>"]
            enc = np.zeros([len(card), len(chars)])
            for vector, char in zip(enc, card):
                vector[chars[char]] = 1.0

            one_hot.append(enc)

        return card_names, one_hot
    


if __name__ == "__main__":
    #data = Datasets.load_pokemon(
    #    upsample=True
    #)

   #names, one_hots = Datasets.load_yugioh()
   #print(names[0])
   #print(one_hots[0])

   data = Datasets.load_pokemon_nlp()