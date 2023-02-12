from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple

from .model import KNRM
from .search import SearchIndex

from langdetect import detect

import torch

from .vectors import Preproc


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
        index_vectors = self.preproc_index([query])
        candidates_idx_matrix = self.index.search(index_vectors)
        candidate_idxs = [i for i in candidates_idx_matrix[0] if i != -1]
        return self.rank_documents(query, candidate_idxs)

    def rank_documents(self, query, candidate_idxs):
        q_vector = self.preproc_knrm([query])[0]
        candidate_vectors = self.document_store.document_vectors[candidate_idxs]

        knrm_queries = torch.LongTensor(
            [q_vector] * len(candidate_vectors))
        knrm_documents = torch.LongTensor(candidate_vectors)
        with torch.no_grad():
            knrm_pred = self.knrm(
                {'query': knrm_queries, 'document': knrm_documents}
            ).reshape(-1)
        sorted_idx = knrm_pred.argsort(descending=True)[
            :self.max_documents_in_suggestion]
        ranked_candidates = [candidate_idxs[i] for i in sorted_idx]

        def make_suggestion_pair(idx):
            doc_id = self.document_store.system_id_to_doc_id[idx]
            return (doc_id, self.document_store.document_src[doc_id])
        return [make_suggestion_pair(idx) for idx in ranked_candidates]


class BuildIndexService:
    def __init__(self, preproc_index: Preproc,
                 preproc_knrm: Preproc,
                 index: SearchIndex) -> None:
        self.preproc_index = preproc_index
        self.preproc_knrm = preproc_knrm
        self.index = index

    def build(self, documents) -> Tuple[int, DocumentStore]:
        texts = []
        # custom implementation, may be replaced by faiss.IndexIDMap
        system_id_to_doc_id = {}
        for system_id, (doc_id, doc) in enumerate(documents.items()):
            texts.append(doc)
            system_id_to_doc_id[system_id] = doc_id

        vectors = self.preproc_index(texts)

        self.index.add(vectors)

        knrm_vectors = self.preproc_knrm(texts)

        return self.index.index.ntotal, DocumentStore(
            document_src=documents,
            system_id_to_doc_id=system_id_to_doc_id,
            document_vectors=np.array(knrm_vectors)
        )
