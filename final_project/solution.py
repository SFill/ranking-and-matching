import json
from typing import Dict, List, Tuple
import numpy as np
import torch

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
    matrix = np.empty((input_dim, out_dim),dtype=np.float32)

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
    assert np.array_equal(matrix[vocab['PAD']],matrix[-2])
    assert np.array_equal(matrix[vocab['OOV']],matrix[-1])
    return matrix.astype(np.float32), vocab


def build_emb_matrixes(glove_file_path: str, knrm_file_path: str, knrm_vocab_path: str):
    # Это должен быть сериализованный файл для инициализации MLP, инициализация происходит примерно вот так:
    # self.mlp.load_state_dict(torch.load(mlp_path))
    # knrm_file_path
    # torch.save(Solution(...).model.embeddings.state_dict())
    glove_embs_matrix, glove_vocab = create_glove_emb_from_file(
        glove_file_path)

    knrm_embs_matrix = torch.load(knrm_file_path)['weight']
    knrm_vocab = json.load(open(knrm_vocab_path, encoding='utf-8'))
    return glove_embs_matrix, glove_vocab, knrm_embs_matrix, knrm_vocab
    # Для векторизации при поиске попробовать два подхода
    # С ембедингами глов
    # С ембедингами глов и knrm, как резерв
    # Для индекса только глов
    # Для кнрм нужно миксануть ембединги
    # import json
    # state_vocab = sol.vocab
    # json.dump(state_vocab, open('vocab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

import string
from collections import Counter
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn

import itertools as it
import re

# Замените пути до директорий и файлов! Можете использовать для локальной отладки.
# При проверке на сервере пути будут изменены
glue_qqp_dir = '/data/QQP/'
glove_path = '/data/glove.6B.50d.txt'

EMB_SIZE = 50


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        # x - [B, L, R]
        numerator = torch.pow(x - self.mu, 2)
        denominator = (2 * self.sigma ** 2)
        r = torch.exp(-numerator / denominator)
        return r


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def load_mlp(self, path):
        self.mlp.load_state_dict(torch.load(path))

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()

        step = 2 / (self.kernel_num - 1)
        ms = np.arange(-1, 1, step).astype(np.float32) + (step/2)
        for mu in ms:
            kernels.append(GaussianKernel(mu, self.sigma))
        kernels.append(GaussianKernel(1, self.exact_sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out




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
        # self.index = faiss.IndexHNSWSQ(EMB_SIZE, faiss.ScalarQuantizer.QT_8bit, 16)
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
            v = self.embedding_matrix[v].sum(axis=0)
            to_stuck.append(v)
        return np.array(to_stuck)


from logging.config import dictConfig
import json
from typing import Dict, List
from flask import Flask
import numpy as np

from flask import request

from langdetect import detect
import torch

import logging

import os

MAX_DOCUMENTS_IN_SUGGESTIONS = 10
KNRM_OUT_LAYERS = []
EMB_SHAPE = (10, 50)





def load_services():

    # ooo_val = 1
    # pad_val = 0
    # vocab = {
    #     f'mama{i}': i for i in range(10)
    # }
    

    # emb_matrix = np.random.random(EMB_SHAPE).astype('float32')

    glove_embs_matrix, glove_vocab, knrm_embs_matrix, knrm_vocab = build_emb_matrixes(
        os.environ['EMB_PATH_GLOVE'],
        os.environ['EMB_PATH_KNRM'],
        os.environ['VOCAB_PATH'],
    )

    
    global_context = {}
    global_context['index'] = SearchIndex(glove_embs_matrix)
    
    global_context['preproc_index'] = Preproc(
        glove_vocab, oov_val=glove_vocab['OOV'], pad_val=glove_vocab['PAD'])
    global_context['preproc_knrm'] = Preproc(
        knrm_vocab, oov_val=knrm_vocab['OOV'], pad_val=knrm_vocab['PAD'])

    knrm = KNRM(knrm_embs_matrix, freeze_embeddings=True, out_layers=KNRM_OUT_LAYERS)
    knrm.load_mlp(os.environ['MLP_PATH'])
    knrm.eval()
    global_context['knrm'] = knrm

    # # тестовый индекс
    # t = "mama1 mama2 mama3"
    # ts = [t]
    # t_vector = global_context['preproc'](ts)
    # print(t_vector)
    # global_context['index'].add(t_vector)

    # global_context['document_matrix_knrm'] = np.array(t_vector).reshape(1, 30)
    # global_context['document_src'] = {0: t, -1: t}
    return global_context


GLOBAL_CONTEXT = load_services()


def response_ok():
    return {'status': "ok"}


def response_error(status):
    return {'status': status}

def ping_view():
    return response_ok()


def query_view():
    index: SearchIndex = GLOBAL_CONTEXT.get('index')
    knrm: KNRM = GLOBAL_CONTEXT.get('knrm')
    preproc_index: Preproc = GLOBAL_CONTEXT.get('preproc_index')
    preproc_knrm: Preproc = GLOBAL_CONTEXT.get('preproc_knrm')
    document_matrix_knrm: np.ndarray = GLOBAL_CONTEXT.get('document_matrix_knrm')
    document_src: Dict[str, str] = GLOBAL_CONTEXT.get('document_src')
    system_id_to_doc_id: Dict[int, str] = GLOBAL_CONTEXT.get(
        'system_id_to_doc_id')

    if not index.is_initialized:
        return response_error('FAISS is not initialized!')
    # content = {"queries": [str]}
    # 1) достали вопросы
    queries = np.array(json.loads(request.json)["queries"])

    # 2) только английские
    lang_check = [
        detect(q) == 'en'
        for q in queries
    ]
    qs_english_idx = np.where(lang_check)[0].tolist()
    qs_english: List[str] = queries[qs_english_idx]

    if len(qs_english) == 0:
        return {
            "lang_check": lang_check,

            # suggestions: [СписокКандидатов[(id_документа,исходный_текст) or None]]
            "suggestions": [None] * len(queries)
        }

    # 3) Получили вектора слов вопросов и вытащили ид кандидатов
    index_vectors = preproc_index(qs_english)
    candidates_idx_matrix = index.search(index_vectors)

    # ранжируем документы
    knrm_vectors = preproc_knrm(qs_english)
    suggestions = [None] * len(queries)
    for english_idx, q_vector, candidate_idxs in zip(qs_english_idx, knrm_vectors, candidates_idx_matrix):
        # вытащили вектора слов кандидатов
        candidate_vectors = document_matrix_knrm[candidate_idxs]
        
        knrm_queries = torch.LongTensor([q_vector] * len(candidate_vectors))
        knrm_documents = torch.LongTensor(candidate_vectors)
        print(knrm_queries.shape, knrm_documents.shape)
        with torch.no_grad():
            knrm_pred = knrm.predict(
                {'query': knrm_queries, 'document': knrm_documents}
            ).reshape(-1)

        sorted_idx = np.argsort(knrm_pred).tolist()[
            ::-1][:MAX_DOCUMENTS_IN_SUGGESTIONS]
        # print(sorted_idx.shape)
        ranked_candidates = candidate_idxs[sorted_idx].tolist()
        # raise Exception([ranked_candidates, sorted_idx,candidate_idxs,type(ranked_candidates)])

        def make_suggestion_pair(idx):
            doc_id = system_id_to_doc_id[idx]
            return (doc_id, document_src[doc_id])
        suggestions[english_idx] = [make_suggestion_pair(idx)
                                    for idx in ranked_candidates]

    # Если документов не хватает, вместо индекса используется -1
    # 1) Если есть -1
    # 2) Если все документы -1
    # В иднексе 70к документов, такова не может быть

    # сделать матрицы
    # query:
    # document:

    # для document нужно хранить тексты
    # взять документы, разбить на вектора, сохранить в матрицу документов

    # сделать query и document
    # собрать вопросы в вектор
    # Достать кандидатов из индекса
    # достать вектора кандидатов из матрицы документов

    # Сделать пары вопрос документ для модели через циклы

    return {
        "lang_check": lang_check,

        # suggestions: [СписокКандидатов[(id_документа,исходный_текст) or None]]
        "suggestions": suggestions

    }


def update_index_view():
    documents: Dict[str, str] = json.loads(request.json)["documents"]

    index: SearchIndex = GLOBAL_CONTEXT.get('index')
    preproc_index: Preproc = GLOBAL_CONTEXT.get('preproc_index')
    preproc_knrm: Preproc = GLOBAL_CONTEXT.get('preproc_knrm')
    index.index.reset()

    # Нужен маппинг текстового ид в ид в системе

    texts = []
    system_id_to_doc_id = {}
    for system_id, (doc_id, doc) in enumerate(documents.items()):
        texts.append(doc)
        system_id_to_doc_id[system_id] = doc_id

    # debug
    system_id_to_doc_id[-1] = '1'


    vectors = preproc_index(texts)
    logger = logging.getLogger("log")
    logger.info(vectors, preproc_index)
    index.add(vectors)
    

    knrm_vectors = preproc_knrm(texts)
    GLOBAL_CONTEXT['document_matrix_knrm'] = np.array(knrm_vectors)
    GLOBAL_CONTEXT['document_src'] = documents
    GLOBAL_CONTEXT['system_id_to_doc_id'] = system_id_to_doc_id

    return {
        'status': 'ok',
        'index_size': index.index.ntotal
    }


def create_app():

    # init code
    app = Flask(__name__)

    app.add_url_rule('/ping', view_func=ping_view, methods=['GET'])
    app.add_url_rule('/query', view_func=query_view, methods=['POST'])
    app.add_url_rule(
        '/update_index', view_func=update_index_view, methods=['POST'])

    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
            'formatter': 'default'
        }},
        'root': {
            'level': 'DEBUG',
            'handlers': ['wsgi']
        }
    })

    return app


app = create_app()
