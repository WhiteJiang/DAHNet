# -*- coding: utf-8 -*-
# @Time    : 2022/10/30
# @Author  : White Jiang
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.corrdatt import CoordAtt


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor=3, dropout=0.5):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class ConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)

        return x


class ChannelMixerConv(nn.Module):
    def __init__(self, num_features, image_size, dropout):
        """
        Args:
            num_features: 通道数 256
            image_size: 图像尺寸
        """
        super().__init__()

        self.mlp = ConvMlp(num_features, hidden_features=num_features * 3, drop=dropout)
        self.image_size = image_size
        self.channel_att = CoordAtt()

    def ChannelGate_forward(self, x):
        residual = x
        BB, HH_WW, CC = x.shape
        HH = WW = int(math.sqrt(HH_WW))
        x = x.permute(0, 2, 1)
        x = x.reshape(BB, CC, HH, WW)
        # b c h w
        x = self.channel_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x
        return x

    def forward(self, x):
        residual = x
        x_pre_norm = self.ChannelGate_forward(x)
        # b h_w c
        x = self.mlp(x_pre_norm, 7, 7)
        return x


class CFM(nn.Module):
    def __init__(self, image_size=7, num_patches=2048, dropout=0.1):
        super().__init__()
        self.channel_mixer = ChannelMixerConv(
            num_patches, image_size, dropout
        )

    def forward(self, x):
        BB, CC, HH, WW = x.shape
        # b h w c
        patches = x.permute(0, 2, 3, 1)
        patches = patches.view(BB, -1, CC)
        # b h*w c
        patches = self.channel_mixer(patches)
        patches = patches.permute(0, 2, 1)
        embedding_rearrange = patches.reshape(BB, CC, HH, WW)
        # embedding_final = embedding_rearrange + x
        return embedding_rearrange


if __name__ == '__main__':
    a = torch.rand((2, 4, 7, 7))
    b = torch.rand((2, 4, 7, 7))
