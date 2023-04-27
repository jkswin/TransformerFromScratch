import numpy as np
from scipy.special import softmax

class Transformer:
    
    def __init__(self) -> None:
        pass


class Encoder:
    pass

    def encode(self):
        embeddings = []
        return embeddings


class Decoder:
    pass 


class Attention:

    def __init__(self, word_embeddings: np.ndarray, weights:np.ndarray) -> None:

        """
        word_embeddings: np.ndarray of dimension [len(word_embedding), n_words]
        weights: weights of dimension [len(word_embedding), len(word_embedding)] 
                where;
                        Q = weights[0]
                        K = weights[1]
                        V = weights[2]
        """
        
        self.word_embeddings = word_embeddings
