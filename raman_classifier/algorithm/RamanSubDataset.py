from typing import List, Dict

from torch.utils.data import Dataset

from .RamanDataset import RamanDataset


class RamanSubDataset(Dataset):

    def __init__(self, base_dataset:RamanDataset, selected_classes:List[int]):
        self.base:RamanDataset = base_dataset
        self.selected_classes:List[int] = sorted(selected_classes)

        self.__global_indices:List[int] = []
        self.id_to_indices:Dict[int, List[int]] = {}

        local_index:int = 0
        for class_id in self.selected_classes:
            global_indices = base_dataset.id_to_indices[class_id]
            local_indices = list(range(local_index, local_index + len(global_indices)))
            self.id_to_indices[class_id] = local_indices
            self.__global_indices.extend(global_indices)
            local_index += len(global_indices)

        self.__class_ids = self.selected_classes

    def __len__(self):
        return len(self.__global_indices)

    def __getitem__(self, local_index):
        global_index = self.__global_indices[local_index]
        return self.base[global_index]

    @property
    def class_ids(self):
        return self.__class_ids
