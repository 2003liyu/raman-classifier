import torch.nn as nn

from .CNNFramework import CNNFramework


class BasicBlockV2(nn.Module):

    expansion:int = 1

    def __init__(self, in_channels:int, out_channels:int, stride:int, dropout_rate:float=0.3):
        nn.Module.__init__(self)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout1d(dropout_rate)

        if in_channels != out_channels * self.expansion or stride != 1:
            self.downsample = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        y = self.bn1(x)
        y = self.relu(y)
        y = self.conv1(y)

        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.dropout(y)

        y += self.downsample(x)

        return y


class ResNetV2(CNNFramework):

    def __init__(self, **kwargs):
        CNNFramework.__init__(self, block_class=BasicBlockV2, **kwargs)