import json
from typing import Dict, List, Tuple
import numpy as np
import torch


import string

import nltk


def read_glove_embeddings(file_path: str) -> Dict[str, List[str]]:
    embedding_data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            current_line = line.rstrip().split(' ')
            embedding_data[current_line[0]] = current_line[1:]
    return embedding_data


def create_glove_emb_from_file(file_path: str,
                               random_seed: int = 0, rand_uni_bound: float = 0.5
                               ) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    glove_emb = read_glove_embeddings(file_path)

    input_dim = len(glove_emb)
    out_dim = len(list(glove_emb.values())[0])
    matrix = np.empty((input_dim, out_dim), dtype=np.float32)

    vocab = dict()
    np.random.seed(random_seed)

    for idx, word in enumerate(glove_emb):
        vocab[word] = idx
        matrix[idx] = glove_emb[word]
    matrix: np.ndarray = np.append(matrix, [
        np.zeros_like(matrix[0]),
        np.random.uniform(-rand_uni_bound, rand_uni_bound, size=out_dim)
    ], axis=0)
    vocab.update({
        'PAD': input_dim, 'OOV': input_dim+1
    })
    assert np.array_equal(matrix[vocab['PAD']], matrix[-2])
    assert np.array_equal(matrix[vocab['OOV']], matrix[-1])
    return matrix.astype(np.float32), vocab


def build_emb_matrixes(glove_file_path: str, knrm_file_path: str, knrm_vocab_path: str):
    glove_embs_matrix, glove_vocab = create_glove_emb_from_file(
        glove_file_path)

    knrm_embs_matrix = torch.load(knrm_file_path)['weight']
    knrm_vocab = json.load(open(knrm_vocab_path, encoding='utf-8'))
    return glove_embs_matrix, glove_vocab, knrm_embs_matrix, knrm_vocab


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
