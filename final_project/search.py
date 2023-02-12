from dataclasses import dataclass
import string
import numpy as np
from typing import Dict, List, Tuple

import faiss
from .model import KNRM
import nltk

from langdetect import detect

import torch



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

    def search(self, vectors: List[List[int]], n_candidates: int = 100) -> np.ndarray:
        vectors = self.idx_vectors_to_doc_vectors(vectors)
        _, I = self.index.search(vectors, self.n_neighbours)
        return I[..., :n_candidates]

    def idx_vectors_to_doc_vectors(self, vectors):
        to_stuck = []
        for v in vectors:
            v = self.embedding_matrix[v].mean(axis=0)
            to_stuck.append(v)
        return np.array(to_stuck)