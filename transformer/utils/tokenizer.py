"""
Adapted from http://ethen8181.github.io/machine-learning/deep_learning/subword/bpe.html
"""

import pandas as pd
from collections import Counter
import re
from tqdm import tqdm
from operator import itemgetter
import json



class Tokenizer:
    def __init__(self, dataset: "list[str]", special_tokens=["<POK>"]) -> None:
        """_summary_

        :param dataset: _description_
        :type dataset: list[str] Assumed to be a list of sentences/paragraphs.
        """
        self.dataset = dataset
        self.vocab = {}
        self.special_tokens = special_tokens
    

class BPE(Tokenizer):

    def __init__(self, dataset: "list[str]"=[], special_tokens=["<POK>"]) -> None:
        super().__init__(dataset, special_tokens)
        self.new_word_token = "</w>"

    def from_file(self, path):
        with open(path, "r") as f:
            bpe_codes = {int(k):tuple(v) for k,v in json.load(f).items()}
            self.bpe_codes = bpe_codes
            self.bpe_ids = {v:k for k,v in bpe_codes.items()}

    def _save(self,save_path):
        with open(save_path, "w") as f:
            json.dump(self.bpe_codes, f)

    
    # Portion of the class for learning merges from corpus


    def _unique_chars(self):
        all_chars = "".join(self.dataset)
        frequencies = Counter(all_chars)
        return list(set(all_chars)), frequencies.most_common()
    
    def _get_character_vocab(self):
        """
        Init the base vocabulary to fit merges:
            1.   Split each sentence by whitespace 
            2.   Split words by characters; separate by whitespace
            3.   Append the new_word character to the end of each word.
            4.   Update self.vocab with unique words as keys and frequency as values.

        e.g. "example" -> "e x a m p l e </w>"
        """
        char_vocab = {}
        for sentence in self.dataset:
            sentence = sentence.split()
            for word in sentence:
                if word in self.special_tokens:
                    pass
                else:    
                    word = " ".join(word.lower()) + f" {self.new_word_token}" 

                if word in char_vocab.keys():
                    char_vocab[word] += 1

                else:
                    char_vocab[word] = 1

        self.vocab = char_vocab
    
    def _get_bigram_stats(self):
        pairs = {}
        for word, frequency in self.vocab.items():
            if word not in self.special_tokens:
                symbols = word.split()
                # count occurrences of pairs
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    current_frequency = pairs.get(pair, 0)
                    pairs[pair] = current_frequency + frequency

        return pairs
    
    def _merge_pairs(self):
        """Step 3. Merge all occurrences of the most frequent pair"""

        bigram_stats = self._get_bigram_stats()
        best_pair = max(bigram_stats, key=bigram_stats.get)
        vocab_out = {}

        # re.escape
        # ensures the characters of our input pair will be handled as is and
        # not get mistreated as special characters in the regular expression.
        pattern = re.escape(' '.join(best_pair))
        replacement = ''.join(best_pair)

        for word_in in self.vocab:
            # replace most frequent pair in all vocabulary
            word_out = re.sub(pattern, replacement, word_in)
            vocab_out[word_out] = self.vocab[word_in]

        self.vocab = vocab_out

        return best_pair



    def fit(self, n_merges = 100, save_path=None):
        """
        https://arxiv.org/pdf/1508.07909.pdf

        Uncases the input text.
        Does not cross word boundaries.
        """

        self.bpe_codes = {}

        unigram_size = len(self._unique_chars()[0])

        self._get_character_vocab()

        for i in tqdm(range(n_merges)):
            best_pair = self._merge_pairs()
            self.bpe_codes[i] = best_pair

        self.bpe_ids = {v:k for k,v in self.bpe_codes.items()}

        vocab_size = unigram_size + len(self.bpe_codes.keys())
        print(f"Initial Vocab Size: {unigram_size}\nFinal Vocab Size: {vocab_size}")

        if save_path:
            self._save(save_path)

    ### Portion of the class for encoding new input 

    def _get_pairs(self, word: "list[str]") -> "set[tuple[str, str]]":
        word = word + [self.new_word_token]
        return set([(word[idx], word[idx+1]) for idx, char in enumerate(word[:-1])])
    
    @staticmethod
    def _create_new_word(word:"list[str]", pair_to_merge):
        first, second = pair_to_merge
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(first)
                i += 1

        return new_word


    def encode(self, sentences: "list[list[str]]") -> "list[list[list[str]]]":
        """
        Tokenizes sentences with 

        :param sentences: Sentences split by whitespace e.g. [["This", "is", "an", "example"]]
        :type sentences: list[list[str]]
        :return: Sentences with additional layer of nesting. Tokenized by merges learned by `self.fit()`: [[['th', 'is</w>'], ['is</w>'], ['a', 'n</w>'], ['ex', 'am', 'p', 'l', 'e</w>']]]
        :rtype: list[list[list[str]]]
        """

        # check model has been fit
        assert self.bpe_codes, "Call fit or specify a model path before trying to encode sentences."
        # check for list of list of strings structure
        assert isinstance(sentences, list) and isinstance(sentences[0], list) and isinstance(sentences[0][0], str)

        bpe_sentences = []
        for sentence in sentences:
            bpe_sentence = []
            for word in sentence:
                word= list(word.lower()) + [self.new_word_token]
                print(word)
                if len(word) == 1:
                    bpe_sentence.append(word)
                    continue

                while True:
                    pairs = self._get_pairs(word)
                    # check if any of the pairs are in the bpe code pairs
                    bpe_codes_pairs = [(pair, self.bpe_ids[pair]) for pair in pairs if pair in self.bpe_ids]
                    print(bpe_codes_pairs)
                    if not bpe_codes_pairs:
                        bpe_sentence.append(word)
                        break
                    # pick the pair with the minimum index in bpe pairs
                    pair_to_merge = min(bpe_codes_pairs, key=itemgetter(1))[0]
                    word = self._create_new_word(word, pair_to_merge)                

            bpe_sentences.append(bpe_sentence)

        return bpe_sentences
    
    def to_ids(self, tokenized_sentences):
        out_sents = []
        for sentence in tokenized_sentences:
            out_sent =[]
            for word in sentence:
                out_word = []
                for subword in word:
                    out_word.append(self.bpe_ids.get(subword, 0))
                out_sent.append(out_word)
            out_sents.append(out_sent)

        return out_sents




