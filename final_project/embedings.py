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
