from typing import Tuple

from .fastkan import FastKANLayer
import torch
from torch import nn

from .BaseNet import BaseNet


class KAN(BaseNet):

    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        self.__n_inputs:int = kwargs["n_inputs"]
        self.__n_outputs:int = kwargs["n_outputs"]
        self.__hidden_sizes:Tuple[int] = kwargs["hidden_sizes"]
        self.model = None

        layers = []
        prev_size = self.__n_inputs
        
        for hidden_size in self.__hidden_sizes:
            layers.append(FastKANLayer(prev_size, hidden_size))
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(FastKANLayer(prev_size, self.__n_outputs))
        self.model = nn.Sequential(*layers)
        self.init_weights()
        
    def __bool__(self)->bool:
        return (self.__n_inputs > 0 and self.__n_outputs > 0)

    @property
    def n_inputs(self)->int:
        return self.__n_inputs
    
    @property
    def n_outputs(self)->int:
        return self.__n_outputs
    
    @property
    def hidden_sizes(self)->Tuple[int]:
        return self.__hidden_sizes
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)
    
    @property
    def title(self)->str:
        return f"{self.base_name}{(self.n_inputs, *self.hidden_sizes, self.n_outputs)}"
