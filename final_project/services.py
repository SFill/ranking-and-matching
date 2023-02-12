from collections import Counter
from dataclasses import dataclass
import logging
import re
import string
import numpy as np
import pandas as pd
from typing import Callable, Dict, List

import faiss
from final_project.model import KNRM
import nltk

from langdetect import detect

import torch


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
    def __init__(self, embedding_matrix: np.ndarray, n_neighbours: int = 10, emb_size: int = 50):
        self.index = faiss.IndexFlatL2(emb_size)
        self.n_neighbours = n_neighbours

        self.embedding_matrix = embedding_matrix
        self.is_initialized = False

    def add(self, vectors: List[List[int]]):
        vectors = self.idx_vectors_to_doc_vectors(vectors)

        self.index.add(vectors)
        self.is_initialized = True
        print(self.index.ntotal)

    def search(self, vectors: List[List[int]], n_candidates: int = 100) -> np.ndarray:
        # index_logger = logging.getLogger('index')
        vectors = self.idx_vectors_to_doc_vectors(vectors)

        # index_logger.info(vectors)
        D, I = self.index.search(vectors, self.n_neighbours)
        return I[..., :n_candidates]

    def idx_vectors_to_doc_vectors(self, vectors):
        to_stuck = []
        for v in vectors:
            v = self.embedding_matrix[v].mean(axis=0)
            to_stuck.append(v)
        return np.array(to_stuck)


@dataclass
class DocumentStore:
    document_src: Dict[str, str]
    system_id_to_doc_id: Dict[int, str]
    document_vectors: np.ndarray


class QueryService:
    max_documents_in_suggestion = 10

    def __init__(self,
                 preproc_index: Preproc,
                 preproc_knrm: Preproc,
                 index: SearchIndex,
                 knrm: KNRM,
                 document_store: DocumentStore,
                 ) -> None:
        self.preproc_index = preproc_index
        self.preproc_knrm = preproc_knrm
        self.index = index
        self.knrm = knrm
        self.document_store = document_store

    def handle_queries(self, queries: List[str]):
        lang_check_list = []
        suggestions_list = []

        for query in queries:
            is_en = detect(query) == 'en'
            lang_check_list.append(is_en)
            if not is_en:
                suggestions_list.append(None)
                continue
            suggestions_list.append(
                self.get_suggestion(query)
            )
        return lang_check_list, suggestions_list

    def get_suggestion(self, query: str):
        # 3) Получили вектора слов вопросов и вытащили ид кандидатов
        index_vectors = self.preproc_index([query])
        candidates_idx_matrix = self.index.search(index_vectors)
        # TODO убрать -1
        candidate_idxs = [i for i in candidates_idx_matrix[0] if i != -1]
        return self.rank_documents(query, candidate_idxs)

    def rank_documents(self, query, candidate_idxs):
        q_vector = self.preproc_knrm([query])[0]
        # ранжируем документы
        # вытащили вектора слов кандидатов
        # candidate_vectors = document_matrix_knrm[candidate_idxs]
        candidate_vectors = self.document_store.document_vectors[candidate_idxs]

        knrm_queries = torch.LongTensor(
            [q_vector] * len(candidate_vectors))
        knrm_documents = torch.LongTensor(candidate_vectors)
        print(knrm_queries.shape, knrm_documents.shape)
        with torch.no_grad():
            knrm_pred = self.knrm(
                {'query': knrm_queries, 'document': knrm_documents}
            ).reshape(-1)

        sorted_idx = np.argsort(knrm_pred).tolist()[
            ::-1][:self.max_documents_in_suggestion]
        # print(sorted_idx.shape)
        ranked_candidates = candidate_idxs[sorted_idx].tolist()
        # raise Exception([ranked_candidates, sorted_idx,candidate_idxs,type(ranked_candidates)])

        def make_suggestion_pair(idx):
            doc_id = self.document_store.system_id_to_doc_id[idx]
            return (doc_id, self.document_store.document_src[doc_id])
        return [make_suggestion_pair(idx) for idx in ranked_candidates]


class BuildeIndexService:
    def __init__(self) -> None:
        pass

    def build(self) -> DocumentStore:
        pass
