import torch.nn as nn
import torch


class SE(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, reduction:int=16):
        nn.Module.__init__(self)
        hidden_dim = max(in_channels // reduction, 4)
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.model(x)