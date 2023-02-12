from math import log2

from torch import Tensor, sort
import torch


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    return (ys_true != ys_pred).sum()


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme =='const':
        return y_value
    elif gain_scheme =="exp2":
        return (2 ** y_value) - 1
    raise ValueError("Invalid argument gain_scheme")


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    _,ind = torch.sort(ys_pred,descending=True)
    ys_ranked = ys_true[ind].double()
    ys_ranked.apply_(lambda x: compute_gain(x,gain_scheme))
    k = len(ys_pred)
    ys_ranked[1:] = ys_ranked[1:]/(torch.arange(2,k+1,dtype=torch.double) + 1).log2()
    return ys_ranked.sum()


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    return dcg(ys_true,ys_pred,gain_scheme)/dcg(ys_true,ys_true,gain_scheme)


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    # допишите ваш код здесь
    _,ind = torch.sort(ys_pred,descending=True)
    ys_ranked = ys_true[ind][:k]
    tp  = ys_ranked.sum()
    if tp == 0:
        return -1
    return tp/len(ys_ranked)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    _,ind = torch.sort(ys_pred,descending=True)
    ys_ranked = ys_true[ind]
    return 1/(ys_ranked.nonzero().item()+ 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    _,ind = torch.sort(ys_pred,descending=True)
    ys_ranked = ys_true[ind]
    
    n = len(ys_pred)
    
    
    p_look = torch.zeros(n)
    p_rel = ys_ranked
    def calc_p_look(i):
        if i == 0:
            p_look[0] = 1
        else:
            p_look[i] = p_look[i-1]*(1-p_rel[i-1])*(1-p_break)

    for i in range(n):
        calc_p_look(i)
    print(p_look)
    return (p_look * p_rel).sum()


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    _,ind = torch.sort(ys_pred,descending=True)
    ys_ranked = ys_true[ind]
    
    def calc_k_p(k):
        k = int(k) +1
        if ys_ranked[k-1] == 0:
            return 0
        return ys_ranked[:k].sum()/(k)
    out = torch.arange(len(ys_true),dtype=torch.double).apply_(calc_k_p)
    return out.sum() / ys_true.sum()
    
