from typing import Union

import torch
from torch.utils.data import Dataset

from .RamanDataset import RamanDataset
from .RamanSubDataset import RamanSubDataset


class ProtoRamanDataset(Dataset):

    def __init__(self, base_dataset:Union[RamanDataset, RamanSubDataset], n_ways:int=5, n_support:int=5, n_query:int=10, n_epsoides:int=100):
        self.base_dataset:Union[RamanDataset, RamanSubDataset] = base_dataset
        self.n_ways:int = n_ways
        self.n_support:int = n_support
        self.n_query:int = n_query
        self.n_epsoides:int = n_epsoides

    def __len__(self)->int:
        return self.n_epsoides

    def __getitem__(self, episode_id:int):
        selected = torch.randperm(len(self.base_dataset.class_ids))[:self.n_ways]
        selected = [self.base_dataset.class_ids[i] for i in selected]

        support_indices, query_indices, y_support, y_query = [], [], [], []
        for new_class_id, class_id in enumerate(selected):
            indices_pool = self.base_dataset.id_to_indices[class_id]
            if len(indices_pool) < self.n_support + self.n_query:
                perm = torch.randint(len(indices_pool), (self.n_support + self.n_query,))
            else:
                perm = torch.randperm(len(indices_pool))[:self.n_support + self.n_query]

            for i, index in enumerate(perm):
                if i < self.n_support:
                    support_indices.append(indices_pool[index])
                    y_support.append(new_class_id)
                else:
                    query_indices.append(indices_pool[index])
                    y_query.append(new_class_id)

        X_support = torch.stack([self.base_dataset[i][0] for i in support_indices])
        X_query = torch.stack([self.base_dataset[i][0] for i in query_indices])
        y_support = torch.tensor(y_support)
        y_query = torch.tensor(y_query)
        return X_support, X_query, y_support, y_query