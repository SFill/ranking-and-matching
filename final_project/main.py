from logging.config import dictConfig
import json
from typing import Dict, List
from flask import Flask
import numpy as np

from flask import request

from .embedings import build_emb_matrixes
from .model import KNRM

from .services import BuildIndexService, DocumentStore, Preproc, QueryService, SearchIndex

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

    knrm = KNRM(knrm_embs_matrix, freeze_embeddings=True,
                out_layers=KNRM_OUT_LAYERS)
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

    index: SearchIndex = GLOBAL_CONTEXT.get('index')
    knrm: KNRM = GLOBAL_CONTEXT.get('knrm')
    preproc_index: Preproc = GLOBAL_CONTEXT.get('preproc_index')
    preproc_knrm: Preproc = GLOBAL_CONTEXT.get('preproc_knrm')
    document_matrix_knrm: np.ndarray = GLOBAL_CONTEXT.get(
        'document_matrix_knrm')
    document_src: Dict[str, str] = GLOBAL_CONTEXT.get('document_src')
    system_id_to_doc_id: Dict[int, str] = GLOBAL_CONTEXT.get(
        'system_id_to_doc_id')

    query_service = QueryService(
        preproc_index=preproc_index,
        preproc_knrm=preproc_knrm,
        index=index
    )

    return global_context


# GLOBAL_CONTEXT = load_services()


def response_ok(attrs: Dict = {}):
    return {'status': "ok", **attrs}


def response_error(status):
    return {'status': status}


class ApplicationWrapper:
    def __init__(self, app: Flask = None) -> None:
        # init flask app
        if app is None:
            app = Flask(__name__)

        app.add_url_rule('/ping', view_func=self.ping_view, methods=['GET'])
        app.add_url_rule('/query', view_func=self.query_view, methods=['POST'])
        app.add_url_rule(
            '/update_index', view_func=self.update_index_view, methods=['POST'])

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
        self.app = app

        # init components
        self.document_store: DocumentStore = None
        self._init_components()

    def _init_components(self):
        glove_embs_matrix, glove_vocab, knrm_embs_matrix, knrm_vocab = build_emb_matrixes(
            os.environ['EMB_PATH_GLOVE'],
            os.environ['EMB_PATH_KNRM'],
            os.environ['VOCAB_PATH'],
        )
        self.index = SearchIndex(glove_embs_matrix)

        self.preproc_index = Preproc(
            glove_vocab, oov_val=glove_vocab['OOV'], pad_val=glove_vocab['PAD'])
        self.preproc_knrm = Preproc(
            knrm_vocab, oov_val=knrm_vocab['OOV'], pad_val=knrm_vocab['PAD'])

        knrm = KNRM(knrm_embs_matrix, freeze_embeddings=True,
                    out_layers=KNRM_OUT_LAYERS)
        knrm.load_mlp(os.environ['MLP_PATH'])
        knrm.eval()
        self.knrm = knrm

    def query_view(self):

        if self.document_store is None:
            return response_error('FAISS is not initialized!')
        # content = {"queries": [str]}
        queries = np.array(request.json["queries"])

        lang_check, suggestions = QueryService(
            preproc_index=self.preproc_index,
            preproc_knrm=self.preproc_knrm,
            index=self.index,
            knrm=self.knrm,
            document_store=self.document_store
        ).handle_queries(queries)

        return {
            "lang_check": lang_check,

            # suggestions: [СписокКандидатов[(id_документа,исходный_текст) or None]]
            "suggestions": suggestions

        }

    def update_index_view(self):
        documents: Dict[str, str] = request.json["documents"]

        self.index.index.reset()
        index_size, self.document_store = BuildIndexService(
            preproc_index=self.preproc_index,
            preproc_knrm=self.preproc_knrm,
            index=self.index
        ).build(documents)

        return response_ok({
            'index_size': index_size
        })

    def ping_view(self):
        return response_ok()


# def update_index_view():
#     documents: Dict[str, str] = request.json["documents"]

#     index: SearchIndex = GLOBAL_CONTEXT.get('index')
#     preproc_index: Preproc = GLOBAL_CONTEXT.get('preproc_index')
#     preproc_knrm: Preproc = GLOBAL_CONTEXT.get('preproc_knrm')
#     index.index.reset()

#     # Нужен маппинг текстового ид в ид в системе

#     texts = []
#     system_id_to_doc_id = {}
#     for system_id, (doc_id, doc) in enumerate(documents.items()):
#         texts.append(doc)
#         system_id_to_doc_id[system_id] = doc_id

#     # debug
#     system_id_to_doc_id[-1] = '1'

#     vectors = preproc_index(texts)

#     index.add(vectors)

#     knrm_vectors = preproc_knrm(texts)
#     GLOBAL_CONTEXT['document_matrix_knrm'] = np.array(knrm_vectors)
#     GLOBAL_CONTEXT['document_src'] = documents
#     GLOBAL_CONTEXT['system_id_to_doc_id'] = system_id_to_doc_id

#     return response_ok({
#         'index_size': index.index.ntotal
#     })


def create_app():
    return ApplicationWrapper().app


app = create_app()
