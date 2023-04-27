import pandas as pd
from collections import Counter



class Tokenizer:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.vocab = []
    
    @staticmethod
    def _unique_chars(dataset: "list[str]"):
        all_chars = "".join(dataset)
        frequencies = Counter(all_chars)
        return list(set(all_chars)), frequencies.most_common()

    def BPE(self, vocab_size = 6000):
        """
        https://arxiv.org/pdf/1508.07909.pdf

        :param vocab_size: _description_, defaults to 6000
        :type vocab_size: int, optional
        """
        base_vocab, frequencies = self._unique_chars(self.dataset)
        



import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
    for i in range(len(symbols)-1):
        pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 8

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

    print(best)
    print(vocab)




if __name__ == "__main__":
    df = pd.read_csv("E:\chat_gpt_tweets\chatgpt_daily_tweets.csv", encoding="utf-8")
    df = df[df["lang"] == "en"]
    dataset = df["text"].to_list()

    tok = Tokenizer(dataset=dataset)
    tok.BPE()



