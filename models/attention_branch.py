from typing import List, Optional, Type

import torch
import torch.nn as nn
from timm.models.byobnet import ByobNet

from utils.utils import module_generator


class BatchNorm2dWithActivation(nn.BatchNorm2d):
    def __init__(self, module):
        super(BatchNorm2dWithActivation, self).__init__(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )

    def forward(self, x):
        self.activations = x.detach().clone()
        return super().forward(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AttentionBranch(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        num_layer: int,
        num_classes: int = 1000,
        inplanes: int = 64,
        multi_task: bool = False,
        num_tasks: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.inplanes = inplanes

        self.layer1 = self._make_layer(
            block, self.inplanes, num_layer, stride=1, down_size=False
        )

        hidden_channel = 10

        self.bn1 = nn.BatchNorm2d(self.inplanes * block.expansion)
        # self.bn1 = BatchNorm2dWithActivation(bn1)
        self.conv1 = conv1x1(self.inplanes * block.expansion, hidden_channel)
        self.relu = nn.ReLU(inplace=True)

        if multi_task:
            if num_tasks is None:
                num_tasks = [1 for _ in range(num_classes)]
            assert num_classes == sum(num_tasks)
            self.conv2 = conv3x3(num_classes, len(num_tasks))
            self.bn2 = nn.BatchNorm2d(len(num_tasks))
        else:
            self.conv2 = conv1x1(hidden_channel, hidden_channel)
            self.bn2 = nn.BatchNorm2d(hidden_channel)
            self.conv3 = conv1x1(hidden_channel, 1)
            self.bn3 = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()

        self.conv4 = conv1x1(hidden_channel, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1)

    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        down_size: bool = True,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        if down_size:
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)
        else:
            inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(inplanes, planes))

            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        self.bn1_activation = x
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)

        self.class_attention = self.sigmoid(x)

        attention = self.conv2(x)
        attention = self.bn2(attention)
        attention = self.relu(attention)

        attention = self.conv3(attention)
        attention = self.bn3(attention)

        self.attention_order = attention
        self.attention = self.sigmoid(attention)

        x = self.conv4(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        return x


class AttentionBranchModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        perception_branch: nn.Module,
        block: Type = Bottleneck,
        num_layer: int = 2,
        num_classes: int = 1000,
        inplanes: int = 64,
        theta_attention: float = 0,
    ) -> None:
        super().__init__()

        self.feature_extractor = feature_extractor

        self.attention_branch = AttentionBranch(block, num_layer, num_classes, inplanes)
        # self.attention_branch = replace_resnet_modules(self.attention_branch)

        self.perception_branch = perception_branch
        self.theta_attention = theta_attention

    def forward(self, x):
        x = self.feature_extractor(x)

        # For Attention Loss
        self.attention_pred = self.attention_branch(x)

        attention = self.attention_branch.attention
        attention = torch.where(
            self.theta_attention < attention, attention.double(), 0.0
        )

        x = x * attention
        x = x.float()
        # x = attention_x + x

        return self.perception_branch(x)


def add_attention_branch(
    base_model: nn.Module,
    division_index: int,
    num_classes: int,
    theta_attention: float = 0,
) -> nn.Module:
    modules = list(base_model.children())
    # [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]

    pre_model = nn.Sequential(*modules[:division_index])
    post_model = nn.Sequential(*modules[division_index:-1], nn.Flatten(1), modules[-1])

    if isinstance(base_model, ByobNet):
        return AttentionBranchModel(
            pre_model,
            post_model,
            Bottleneck,
            2,
            num_classes,
            inplanes=64,
            theta_attention=theta_attention,
        )

    final_layer = module_generator(modules[division_index], reverse=True)
    for module in final_layer:
        if isinstance(module, nn.modules.batchnorm._NormBase):
            inplanes = module.num_features // 2
            break
        elif isinstance(module, nn.modules.conv._ConvNd):
            inplanes = module.out_channels
            break

    return AttentionBranchModel(
        pre_model,
        post_model,
        Bottleneck,
        2,
        num_classes,
        inplanes=inplanes,
        theta_attention=theta_attention,
    )
