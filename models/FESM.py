# -*- coding: utf-8 -*-
# @Time    : 2022/11/8
# @Author  : White Jiang
import math

import torch.nn as nn
import torch


def get_list(w, k):
    width_list = []
    for i in range(1, math.ceil(w / k) + 1):
        if i == math.ceil(w / k):
            width_list.append(w - k * (i - 1))
        else:
            width_list.append(w // k)
    print(width_list)


class FESM(nn.Module):
    def __init__(self, in_channel=1024, k=2):
        super(FESM, self).__init__()
        self.k = k
        self.stripconv = nn.Sequential(
            # in_channel out_channel kernel_size stride padding
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.stripconv_h = nn.Sequential(
            # in_channel out_channel kernel_size stride padding
            nn.Conv2d(in_channel, 1, 1, 1, 0),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm):
        # print(fm[0][0])
        b, c, w, h = fm.shape
        # 宽度方向上划分 k b c w/k h
        fm_w = torch.split(fm, w // self.k, dim=2)
        fm_h = torch.split(fm, h // self.k, dim=3)  # 高度方向划分
        # k b 1 w/k h
        fmw_conv = map(self.stripconv, fm_w)
        fmh_conv = map(self.stripconv_h, fm_h)
        # k b 1 1 1
        fmw_pool = list(map(self.avgpool, fmw_conv))
        fmh_pool = list(map(self.avgpool, fmh_conv))
        # print(fmh_pool[0].size())
        # k b 1 1 1 -> b 1 k 1
        fmw_pool = torch.cat(fmw_pool, dim=2)
        fmh_pool = torch.cat(fmh_pool, dim=3)
        # # soft计算用于注意力权重
        fmw_softmax = torch.softmax(fmw_pool, dim=2)  # every parts has one score [B*C*K*1]
        fmw_softmax = torch.repeat_interleave(fmw_softmax, w // self.k, dim=2)
        # print('fmw_softmax', fmw_softmax)

        fmh_softmax = torch.softmax(fmh_pool, dim=3)
        fmh_softmax = torch.repeat_interleave(fmh_softmax, w // self.k, dim=3)
        # b c h w
        fm_final = fmw_softmax * fmh_softmax

        alpha = 0.5
        # 输入特征 加上增强的特征

        # print(fm[0][0])
        fms_boost = fm + alpha * (fm * fm_final)
        # print(fms_boost[0][0])
        beta = 0.5

        # # 将小于最大值的置为1 等于的置数为beat 对应论文的1 - beta
        fms_softmax_suppress = []
        for index, var in enumerate(fm_final):
            fms_max = torch.max(var)
            temp = torch.clamp((fm_final[index] < fms_max * 0.95).float(), min=beta)
            fms_softmax_suppress.append(temp)
        fms_softmax_suppress = torch.stack(fms_softmax_suppress)
        # # 抑制后的特征
        fms_suppress = fm * fms_softmax_suppress
        # print(fms_suppress[0][0])
        return fms_boost, fms_suppress