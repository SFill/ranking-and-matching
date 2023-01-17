from logging.config import dictConfig
import json
from typing import Dict, List
from flask import Flask
import numpy as np

from flask import request

from .embedings import build_emb_matrixes
from .model import KNRM

from .services import Preproc, SearchIndex

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
    queries = np.array(request.json["queries"])

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
    documents: Dict[str, str] = request.json["documents"]

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
