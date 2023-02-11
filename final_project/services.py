from collections import Counter
import logging
import re
import string
import numpy as np
import pandas as pd
from typing import Callable, Dict, List

import faiss
import nltk


EMB_SIZE = 50




# def _filter_rare_words(vocab: Dict[str, int], min_occurancies: int) -> Dict[str, int]:
#     result = []
#     for el,occr in vocab.items():
#         if occr >= min_occurancies:
#             result.append(el)
#     return result


# def get_all_tokens(list_of_df: List[pd.DataFrame], min_occurancies: int) -> List[str]:
#     uniq_txt = set()
#     key_left = 'text_left'
#     key_right = 'text_right'

#     for df in list_of_df:
#         for txt in df[key_left].values:
#             uniq_txt.add(txt)
#         for txt in df[key_right].values:
#             uniq_txt.add(txt)
#     tokens = []
#     for txt in uniq_txt:
#         tokens.extend(simple_preproc(txt))
#     c = Counter(tokens)
#     return _filter_rare_words(c, min_occurancies)


class Preproc:
    def __init__(self, vocab: Dict[str, int], oov_val: int, pad_val: int, vector_len: int = 30):
        self.vocab = vocab
        self.oov_val = oov_val
        self.vector_len = vector_len
        self.pad_val = pad_val

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        vector = []
        tokenized_text = tokenized_text[:self.vector_len]
        for t in tokenized_text:
            vector.append(
                self.vocab.get(t, self.oov_val)
            )
        return vector

    def _convert_text_idx_to_token_idxs(self, txts: List[str]) -> List[List[int]]:
        # допишите ваш код здесь
        result = []
        for txt in txts:
            tokens = self.preproc_func(txt)
            vector_of_token_idxs = self._tokenized_text_to_index(tokens)
            vector_of_token_idxs += [self.pad_val] * \
                (self.vector_len - len(vector_of_token_idxs))
            result.append(vector_of_token_idxs)
        return result

    def __call__(self, txts):
        return self._convert_text_idx_to_token_idxs(txts)

    def preproc_func(self, inp_str):
        def handle_punctuation(s, repl=" "):
            sr = ''
            for l in s:
                if l in string.punctuation:
                    l = repl
                sr += l
            return sr
        inp_str = handle_punctuation(inp_str).lower()

        tokens = nltk.word_tokenize(inp_str)
        return tokens


class SearchIndex:
    def __init__(self, embedding_matrix: np.ndarray, n_neighbours: int = 10):
        # self.index = faiss.IndexFlatL2(EMB_SIZE)
        self.index = faiss.IndexHNSWFlat(EMB_SIZE, 32)
        if not self.index.is_trained:
            # тренировка
            pass
        self.n_neighbours = n_neighbours

        self.embedding_matrix = embedding_matrix
        self.is_initialized = False

    def add(self, vectors: List[List[int]]):
        vectors = self.idx_vectors_to_doc_vectors(vectors)

        self.index.add(vectors)
        self.is_initialized = True
        print(self.index.ntotal)

    def search(self, vectors: List[List[int]], n_candidates: int = 20) -> np.ndarray:
        # index_logger = logging.getLogger('index')
        vectors = self.idx_vectors_to_doc_vectors(vectors)
        
        # index_logger.info(vectors)
        D, I = self.index.search(vectors, self.n_neighbours)
        # I = self.index.search(vectors, self.n_neighbours)
        # print(I)
        return I[..., :n_candidates]

    def idx_vectors_to_doc_vectors(self, vectors):
        to_stuck = []
        for v in vectors:
            v = self.embedding_matrix[v].mean(axis=0)
            to_stuck.append(v)
        return np.array(to_stuck)

