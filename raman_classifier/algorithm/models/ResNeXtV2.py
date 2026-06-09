import torch.nn as nn

from .CNNFramework import CNNFramework


class BottleneckV2(nn.Module):

    expansion:int = 2

    def __init__(self, in_channels:int, out_channels:int, stride:int, groups:int=32, base_width:int=4, dropout_rate:float=0.3):
        nn.Module.__init__(self)

        width = int(out_channels * (base_width / 64.0)) * groups

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

        self.bn3 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
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

        y = self.bn3(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.dropout(y)

        y += self.downsample(x)

        return y


class ResNeXtV2(CNNFramework):

    def __init__(self, **kwargs):
        CNNFramework.__init__(self, block_class=BottleneckV2, **kwargs)