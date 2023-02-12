import math

import numpy as np

import torch
from torch import nn


from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List

# from sklearn.
from sklearn.preprocessing import StandardScaler

from torch import Tensor, sort



def listnet_ce_loss(y_i,z_i):
    P_y_i = torch.softmax(y_i,dim=0)
    P_z_i = torch.softmax(z_i,dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i))


def listnet_kl_loss(y_i,z_i):
    P_y_i = torch.softmax(y_i,dim=0)
    P_z_i = torch.softmax(z_i,dim=0)
    return -torch.sum(P_y_i * torch.log(P_z_i/P_y_i))


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme =='const':
        return y_value
    elif gain_scheme =="exp2":
        return (2 ** y_value) - 1
    raise ValueError("Invalid argument gain_scheme")


def dcg_k(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str, k:int) -> float:
    k = min(k,len(ys_true))
    
    ys_true = ys_true.reshape(-1)
    ys_pred = ys_pred.reshape(-1)
    
    ind = torch.argsort(ys_pred,descending=True,dim=0)
    ys_ranked = ys_true[ind][:k].double()
    # print(ys_ranked)
    ys_ranked.apply_(lambda x: compute_gain(x,gain_scheme))
    # print(ys_ranked)
    disc = ys_ranked/(torch.arange(k,dtype=torch.double)+2).log2()
    # print(disc)
    return disc.sum()



def ndcg_k(ys_true: Tensor, ys_pred: Tensor, k:int) -> float:
    gain_scheme = 'exp2'
    # print(dcg_k(ys_true,ys_pred,gain_scheme,k),dcg_k(ys_true,ys_true,gain_scheme,k))
    return dcg_k(ys_true,ys_pred,gain_scheme,k)/dcg_k(ys_true,ys_true,gain_scheme,k)


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # укажите архитектуру простой модели здесь
        self.model = nn.Sequential(
            nn.Linear(num_input_features, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim,1)
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    BATCH_SIZE = 60
    
    
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        ys_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        ys_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, ys_train, query_ids_train, X_test, ys_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, ys_train, self.query_ids_train,
            X_test, ys_test, self.query_ids_test) = self._get_data()
        # допишите ваш код здесь
        X_train_scaled = self._scale_features_in_query_groups(X_train,self.query_ids_train)
        X_test_scaled = self._scale_features_in_query_groups(X_test,self.query_ids_test)
        
        self.X_train= torch.FloatTensor(X_train_scaled)
        self.X_test = torch.FloatTensor(X_test_scaled)
        
        self.ys_train = torch.FloatTensor(ys_train)
        self.ys_test = torch.FloatTensor(ys_test)
        print(ys_train.shape,ys_test.shape)
        
        # N_train = 1000
        # N_valid = 500
        # vector_dim = 100
        # self.X_train, self.X_test, self.ys_train, self.ys_test = make_dataset(N_train, N_valid, vector_dim)
        
        
        

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        uniq_qs = np.unique(inp_query_ids)
        out_array = np.zeros(inp_feat_array.shape)
        for id_ in uniq_qs:
            index = inp_query_ids == id_
            batch = inp_feat_array[index]
            tr = StandardScaler()
            batch_out = tr.fit_transform(batch)
            out_array[index] = batch_out
        return out_array
        # return inp_feat_array
            
            
        
        

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        # допишите ваш код здесь
        net = ListNet(listnet_num_input_features,listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        # допишите ваш код здесь
        result = []
        for epoch in range(self.n_epochs):
            print(f"epoch: {epoch + 1}; train")
            self._train_one_epoch()
            print(f"epoch: {epoch + 1}; eval")
            ndcg = self._eval_test_set()
            print(f"epoch: {epoch + 1}; ndcg: {ndcg}")
            result.append(ndcg)
        return result
            # print(f"epoch: {epoch + 1}.\tNumber of swapped pairs: " 
            #       f"{valid_swapped_pairs}/{N_valid * (N_valid - 1) // 2}\t"
            #       f"nDCG: {ndcg_score:.4f}")
            

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # CE
        # print(batch_ys.shape,batch_pred.shape)
        return listnet_kl_loss(batch_ys,batch_pred)
    

    def _train_one_epoch(self) -> None:
        self.model.train()
        avg_loss = []
        for q_id in np.unique(self.query_ids_train):
            select_index = self.query_ids_train == q_id
            query_X = self.X_train[select_index]
            query_y = self.ys_train[select_index]
        # for _ in range(1) :
        #     # select_index = self.query_ids_train == q_id
        #     # query_X = self.X_train[select_index]
        #     # query_y = self.ys_train[select_index]
        #     query_X = self.X_train
        #     query_y = self.ys_train
            
            
#             self.optimizer.zero_grad()
#             query_pred = self.model(query_X)
#             batch_loss = self._calc_loss(query_y,query_pred)
#             # print()
#             # print(batch_loss)
#             batch_loss.backward(retain_graph=True)
#             self.optimizer.step()
            

            # print(q_id,batch_loss)
                    
            
            shuffle_index = torch.randperm(len(query_X))
            query_X = query_X[shuffle_index]
            query_y = query_y[shuffle_index]
            
            cur_batch = 0 
            for it in range(len(query_X)//self.BATCH_SIZE):
                batch_X = query_X[cur_batch:cur_batch+self.BATCH_SIZE] 
                batch_y = query_y[cur_batch:cur_batch+self.BATCH_SIZE] 
                cur_batch+= self.BATCH_SIZE

                self.optimizer.zero_grad()
                batch_pred = self.model(batch_X).reshape(-1)
                batch_loss = self._calc_loss(batch_y,batch_pred)
                # print()
                # print(batch_loss)
                batch_loss.backward(retain_graph=True)
                self.optimizer.step()
                # if len(batch_X) > 0:
                #     # print(self.model)
                #     # print(batch_X.shape)
                #     batch_pred = self.model(batch_X)
                #     batch_loss = self._calc_loss(batch_y,batch_pred)
                #     # print()
                #     # print(batch_loss)
                #     batch_loss.backward(retain_graph=True)
                #     self.optimizer.step()
                    
                    # print(f"Model structure: {self.model}\n\n")
                    # for name, param in self.model.named_parameters():
                    #     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
                    #     break
            # avg_loss.append(batch_loss.detach().numpy())
        # print(np.mean(avg_loss))
                    
                    
            # raise Exception
            # print(q_id,batch_loss)
        

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            # допишите ваш код здесь
            for id_ in np.unique(self.query_ids_test):
                index = self.query_ids_test == id_
                query_X = self.X_test[index]
                query_y_true = self.ys_test[index]
            # for _ in range(1):
            #     query_X = self.X_test
            #     query_y_true = self.ys_test
                
                
                query_y_pred = self.model(query_X)
                # print(query_y_true[:2],query_y_pred[:2])
                # print(query_y_true,query_y_pred)
                ndcgs.append(
                    self._ndcg_k(query_y_true, query_y_pred, self.ndcg_top_k)
                )
            # print(ndcgs)
            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        # допишите ваш код здесь
        return ndcg_k(ys_true,ys_pred, ndcg_top_k)
    
    
    

