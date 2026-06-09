from typing import Tuple, List
import torch.nn as nn
import torch
import torch.nn.functional as F
from .BaseNet import BaseNet


class CNNFramework(BaseNet):
    def __init__(self, **kwargs):
        BaseNet.__init__(self, **kwargs)
        block_class = kwargs.get("block_class")
        self.__n_outputs = kwargs.get("n_outputs", 0)
        self.__hidden_sizes = kwargs.get("hidden_sizes", (16, 32, 32))


        self.stem = nn.Sequential(
            nn.Conv1d(1, self.__hidden_sizes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(self.__hidden_sizes[0]),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        stages = []
        prev_size = self.__hidden_sizes[0]
        for i, hidden_size in enumerate(self.__hidden_sizes[1:]):

            block = block_class(in_channels=prev_size, out_channels=hidden_size, stride=2, dropout_rate=0.2)
            prev_size = block.expansion * hidden_size
            stages.append(block)
        self.body = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(prev_size, prev_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(prev_size // 2, self.__n_outputs)
        )


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.Linear):

            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        x = x.float()


        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)


        elif x.dim() == 2:
            x = x.unsqueeze(1)


        elif x.dim() == 3:
            if x.shape[1] != 1:
                x = x.transpose(1, 2)



        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)

        return y

    @property
    def title(self) -> str:
        h_str = "-".join(map(str, self.__hidden_sizes))
        return f"CNN_{h_str}"

    @property
    def n_outputs(self) -> int:
        return self.__n_outputs

    @property
    def hidden_sizes(self) -> Tuple[int]:
        return self.__hidden_sizes