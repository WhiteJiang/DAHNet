from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.FESM import FESM
from models.CFM import CFM
from models.mcm import MCM


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 输出的中间通道数
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_backbone(pretrained=True, progress=True, **kwargs):
    model = ResNet_Backbone(Bottleneck, [3, 4, 6], **kwargs)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        for name in list(state_dict.keys()):
            if 'fc' in name or 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict)
    return model


class TransLayer(nn.Module):
    def __init__(self, block):
        super(TransLayer, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.inplanes = 1024
        self.groups = 1
        self.base_width = 64
        self.layer4 = self._make_layer(block, 512, stride=2)

    def _make_layer(self, block, planes, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer4(x)
        return out


class Trans_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(Trans_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
            if _ == 1 and self.is_local:
                layers.append(CFM())
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_Refine(nn.Module):

    def __init__(self, block, layer, is_local=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet_Refine, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 1024
        self.dilation = 1
        self.is_local = is_local
        self.groups = groups
        self.base_width = width_per_group
        self.layer4 = self._make_layer(block, 512, layer, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer4(x)

        pool_x = self.avgpool(x)
        pool_x = torch.flatten(pool_x, 1)
        if self.is_local:
            return x, pool_x
        else:
            return pool_x

    def forward(self, x):
        return self._forward_impl(x)


def SEMICON_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = ResNet_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        for name in list(state_dict.keys()):
            if not 'layer4' in name:
                state_dict.pop(name)
        model.load_state_dict(state_dict, strict=False)
    return model


def Trans(pretrained=True):
    model = TransLayer(Bottleneck)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.0' in name:
                # print(name)
                pretrain_keys.append(name)
        for key in pretrain_keys:
            model.state_dict()[key].copy_(state_dict[key])
        # model.load_state_dict(state_dict, strict=False)
    # print(model.state_dict())
    return model


def Trans_refine(is_local=True, pretrained=True, progress=True, **kwargs):
    model = Trans_Refine(Bottleneck, 3, is_local, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/jx/code/SEMICON/resnet50.pth')
        pretrain_keys = []
        for name in list(state_dict.keys()):
            if 'layer4.1' in name or 'layer4.2' in name:
                # if 'layer4' in name:
                pretrain_keys.append(name)
        for key in pretrain_keys:
            key2 = list(key)
            if int(key2[7]) == 1:
                key2[7] = '{}'.format(int(key2[7]) - 1)
            # key2[7] = '{}'.format(int(key2[7]) - 1)
            key2 = ''.join(key2)
            model.state_dict()[key2].copy_(state_dict[key])
    # print(model.state_dict())
    return model


"""
Visual
"""


class DAHNET(nn.Module):
    def __init__(self, code_length=12, num_classes=200, feat_size=2048, device='cpu', pretrained=False):
        super(DAHNET, self).__init__()
        # 骨干网
        self.backbone = SEMICON_backbone(pretrained=pretrained)

        self.attention = FESM()
        self.trans = Trans(pretrained=pretrained)
        self.mcm = MCM()

        self.refine_global = SEMICON_refine(is_local=False, pretrained=pretrained)
        self.refine_local = Trans_refine(pretrained=pretrained)

        self.cls = nn.Linear(feat_size, num_classes)
        self.cls_loc = nn.Linear(feat_size, num_classes)

        self.hash_layer_active = nn.Sequential(
            nn.Tanh(),
        )
        self.code_length = code_length

        # global
        if self.code_length != 32:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2, feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))
        else:
            self.W_G = nn.Parameter(torch.Tensor(code_length // 2 + 1, feat_size))
            torch.nn.init.kaiming_uniform_(self.W_G, a=math.sqrt(5))

            # local
        self.W_L1 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L1, a=math.sqrt(5))
        self.W_L2 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L2, a=math.sqrt(5))
        self.W_L3 = nn.Parameter(torch.Tensor(code_length // 6, feat_size))
        torch.nn.init.kaiming_uniform_(self.W_L3, a=math.sqrt(5))

        self.bernoulli = torch.distributions.Bernoulli(0.5)
        self.device = device

    def forward(self, x):
        out = self.backbone(x)  # .detach()
        global_f = self.refine_global(out)
        # for local
        feature_boost1, fms_suppress1 = self.attention(out)
        feature_boost2, fms_suppress2 = self.attention(fms_suppress1)
        feature_boost3, _ = self.attention(fms_suppress2)

        out_local1 = self.trans(feature_boost1)
        out_local2 = self.trans(feature_boost2)
        out_local3 = self.trans(feature_boost3)
        # MCM
        f2_from_f1, f1_from_f2 = self.mcm(out_local2, out_local1)
        f3_from_f2, f2_from_f3 = self.mcm(out_local3, out_local2)

        out_local1 = out_local1 + f1_from_f2
        out_local2 = out_local2 + 0.5 * (f2_from_f1 + f2_from_f3)
        out_local3 = out_local3 + f3_from_f2
        #
        local_f1, avg_local_f1 = self.refine_local(out_local1)
        local_f2, avg_local_f2 = self.refine_local(out_local2)
        local_f3, avg_local_f3 = self.refine_local(out_local3)

        deep_S_G = F.linear(global_f, self.W_G)

        deep_S_1 = F.linear(avg_local_f1, self.W_L1)
        deep_S_2 = F.linear(avg_local_f2, self.W_L2)
        deep_S_3 = F.linear(avg_local_f3, self.W_L3)

        deep_S = torch.cat([deep_S_G, deep_S_1, deep_S_2, deep_S_3], dim=1)
        # 哈希激活层
        ret = self.hash_layer_active(deep_S)
        if self.training:
            cls = self.cls(global_f)
            cls1 = self.cls_loc(avg_local_f1)
            cls2 = self.cls_loc(avg_local_f2)
            cls3 = self.cls_loc(avg_local_f3)
            return ret, local_f1, cls, cls1, cls2, cls3
        return ret, local_f1


def dahnet(code_length=12, num_classes=200, feat_size=2048, device='cpu', pretrained=False, **kwargs):
    # 实际att_size = 1
    model = DAHNET(code_length, num_classes, feat_size, device, pretrained, **kwargs)
    return model
