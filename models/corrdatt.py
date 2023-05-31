# -*- coding: utf-8 -*-
# @Time    : 2023/5/10
# @Author  : White Jiang
import torch
import torch.nn as nn


class sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class swish(nn.Module):
    def __init__(self, inplace=True):
        super(swish, self).__init__()
        self.sigmoid = sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp=2048, oup=2048, groups=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = swish()

    def forward(self, x):
        identity = x
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        y = torch.mul(x_h, x_w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y).sigmoid()

        y = identity * y

        return y
