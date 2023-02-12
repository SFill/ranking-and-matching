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

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
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
