import torch.nn as nn

from .CNNFramework import CNNFramework


class BasicBlock(nn.Module):

    expansion:int = 1

    def __init__(self, in_channels:int, out_channels:int, stride:int, dropout_rate:float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(dropout_rate)

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.dropout(y)

        y += self.downsample(x)
        y = self.relu(y)

        return y


class ResNet(CNNFramework):

    def __init__(self, **kwargs):
        CNNFramework.__init__(self, block_class=BasicBlock, **kwargs)