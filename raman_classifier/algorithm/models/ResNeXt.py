# import torch.nn as nn
#
# from .CNNFramework import CNNFramework
#
#
# class Bottleneck(nn.Module):
#
#     expansion:int = 2
#
#     def __init__(self, in_channels:int, out_channels:int, stride:int, groups:int=32, base_width:int=4, dropout_rate:float=0.3):
#         nn.Module.__init__(self)
#
#         width = int(out_channels * (base_width / 64.0)) * groups
#
#         self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(width)
#
#         self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
#         self.bn2 = nn.BatchNorm1d(width)
#
#         self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
#         self.dropout = nn.Dropout1d(dropout_rate)
#
#         if in_channels != out_channels or stride != 1:
#             self.downsample = nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False)
#         else:
#             self.downsample = nn.Identity()
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         y = self.conv1(x)
#         y = self.bn1(y)
#         y = self.relu(y)
#
#         y = self.conv2(y)
#         y = self.bn2(y)
#         y = self.relu(y)
#
#         y = self.conv3(y)
#         y = self.bn3(y)
#         y = self.dropout(y)
#
#         y += self.downsample(x)
#         y = self.relu(y)
#
#         return y
#
#
# class ResNeXt(CNNFramework):
#
#     def __init__(self, **kwargs):
#         CNNFramework.__init__(self, block_class=Bottleneck, **kwargs)


import torch.nn as nn
from .CNNFramework import CNNFramework


class Bottleneck(nn.Module):
    expansion: int = 2

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 groups: int = 8, base_width: int = 16, dropout_rate: float = 0.3):
        nn.Module.__init__(self)

        # 核心：计算宽度时，必须保证能被 groups 整除
        width = int(out_channels * (base_width / 64.0)) * groups

        # 第一层：1x1
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)

        # 第二层：3x3 组卷积
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm1d(width)

        # 第三层：1x1 还原
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.dropout = nn.Dropout1d(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

        # 核心修复：这里是 Size Mismatch 的重灾区
        # 必须确保 downsample 的输出通道数 严格等于 out_channels * expansion
        if in_channels != out_channels * self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)

        out += identity  # 此时维度强制对齐
        return self.relu(out)


class ResNeXt(CNNFramework):
    def __init__(self, **kwargs):
        # 强制在这里指定参数，确保不论 API 怎么传，模型结构是稳定的
        kwargs["groups"] = 8
        kwargs["base_width"] = 16
        super(ResNeXt, self).__init__(block_class=Bottleneck, **kwargs)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        return super().forward(x)