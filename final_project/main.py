from logging.config import dictConfig
from typing import Dict
from flask import Flask
import numpy as np

from flask import request

from .vectors import build_emb_matrixes, Preproc
from .model import KNRM

from .services import BuildIndexService, DocumentStore, Preproc, QueryService
from .search import SearchIndex
import os


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
                    out_layers=[])
        knrm.load_mlp(os.environ['MLP_PATH'])
        knrm.eval()
        self.knrm = knrm

    def query_view(self):

        if self.document_store is None:
            return response_error('FAISS is not initialized!')
        # request.json = {"queries": [str]}
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

            # suggestions: [CandidateList[(document_id,document) or None]]
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


def create_app():
    return ApplicationWrapper().app


app = create_app()
