import unittest
from toy_data.dataset_wrapper import Datasets
from utils.tokenizer import BPE

class TestTokenizer(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.data = Datasets.load_pokemon_nlp()
        pokemon_text = [p["desc"] for p in self.data]
        self.pok_tokenizer = BPE(pokemon_text, special_tokens=["<POK>"])

        self.BPE()
        
    @unittest.SkipTest
    def test_BPE(self):
        save_path="bpe_model.json"
        self.tokenizer.fit(save_path=save_path, n_merges=100)
        #tokenizer.from_file(path=save_path)
        subwords = self.tokenizer.encode([["This", "is", "brown", "example"]])
        ids = self.tokenizer.to_ids(subwords)
        print(subwords)
        print(ids)
        print(self.tokenizer.bpe_codes)

    def test_get_character_vocab(self):
        # initializes the vocabulary by splitting 
        self.tokenizer._get_character_vocab()
        print(self.tokenizer.vocab)
    


if __name__ == "__main__":
    unittest.main()