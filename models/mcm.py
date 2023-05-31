# -*- coding: utf-8 -*-
# @Time    : 2023/5/4
# @Author  : White Jiang
import torch.nn
import torch.nn as nn


class MCM(nn.Module):
    def __init__(self):
        super(MCM, self).__init__()
        self.soft = torch.nn.Softmax(dim=2)

    def forward(self, f1, f2):
        """
        f1:high-level
        f2:low-level
        """
        similarity = torch.cosine_similarity(f1, f2, dim=1)
        b, c, h, w = f1.size()
        similarity = similarity.view(b, 1, h * w)
        similarity = self.soft(similarity)
        similarity_1 = 1.0 - similarity
        similarity_1 = self.soft(similarity_1)
        similarity = similarity.view(b, 1, h, w)
        similarity_1 = similarity_1.view(b, 1, h, w)
        # print(similarity.size())
        f1_from_f2 = f2 * similarity
        f2_from_f1 = f1 * similarity_1
        return f1_from_f2, f2_from_f1
