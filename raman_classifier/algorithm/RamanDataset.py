from typing import List, Dict, Tuple, Optional

import psutil
import tables

import torch
from torch.utils.data import Dataset

from .ProjectFile import ProjectFile


class RamanDataset(Dataset):


    def __init__(self, project_file, is_deep, is_proto, is_query=False):
        self.is_query = is_query
        self.__is_deep:bool = is_deep
        self.__is_proto:bool = is_proto
        self.__project_file:ProjectFile = project_file
        self.__class_ids:List[int] = []
        self.__id_to_indices:Dict[int, List[int]] = {}
        self.__index_to_row_node:Dict[int, tables.Group] = {}
        self.__sum_rows:int = 0
        self.__Y:Optional[torch.Tensor] = None
        self.__X:Optional[torch.Tensor] = None
        self.update()
        
    def __len__(self)->int:
        return self.__sum_rows
    
    def __getitem__(self, index:int)->Tuple[torch.Tensor, torch.Tensor]:

        if self.is_deep:
            key = "normalized_data"
        else:
            key = "proto_pca_features" if self.is_proto else "pca_features"


        if self.__X is not None:
            x:torch.Tensor = self.__X[index]
        else:
            row_node = self.__index_to_row_node[index]
            x:torch.Tensor = torch.tensor(row_node[key], dtype=torch.float32)

        y:torch.Tensor = self.__Y[index]



        return x, y
    
    @property
    def is_proto(self)->bool:
        return self.__is_proto
    
    @property
    def is_deep(self)->bool:
        return self.__is_deep

    def update(self):
        self.__class_ids.clear()
        self.__id_to_indices.clear()
        self.__index_to_row_node.clear()
        self.__sum_rows:int = 0
        self.__Y = None
        self.__X = None

        Y = []
        train_classes:tables.Group = (self.__project_file.train_classes if not self.is_proto else self.__project_file.proto_train_classes)
        for class_node in train_classes:
            class_id:int = class_node._v_attrs.class_id
            self.__class_ids.append(class_id)

            indices = []
            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                Y.append(torch.tensor(class_id, dtype=torch.long))

                self.__index_to_row_node[self.__sum_rows] = row_node
                indices.append(self.__sum_rows)
                self.__sum_rows += 1

            self.__id_to_indices[class_id] = indices

        if Y:
            self.__Y = torch.stack(Y)
        else:
            self.__Y = None

        if not self.is_deep:
            key:str = "pca_features" if not self.is_proto else "proto_pca_features"
        else:
            key:str = "normalized_data"

        mem = psutil.virtual_memory()
        if 4 * self.n_features * self.__sum_rows < 0.5*mem.available:
            X = []
            for class_node in train_classes:
                for row_node in class_node:
                    if row_node._v_name == "protos":
                        continue

                    X.append(torch.tensor(row_node[key], dtype=torch.float32))

            if X:
                self.__X = torch.stack(X)
            else:
                self.__X = None

    @property
    def n_features(self)->int:
        return self.__project_file.n_features

    @property
    def id_to_indices(self)->Dict[int, List[int]]:
        return self.__id_to_indices
    
    @property
    def class_ids(self)->List[int]:
        return self.__class_ids
    