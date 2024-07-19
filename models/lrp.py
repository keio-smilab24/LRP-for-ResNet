import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock as OriginalBasicBlock
from torchvision.models.resnet import Bottleneck as OriginalBottleneck

from models.attention_branch import BasicBlock as ABNBasicBlock
from models.attention_branch import Bottleneck as ABNBottleneck


class LinearWithActivation(nn.Linear):
    def __init__(self, module, out_features=None):
        in_features = module.in_features
        if out_features is None:
            out_features = module.out_features
        bias = module.bias is not None
        super(LinearWithActivation, self).__init__(in_features, out_features, bias)
        self.activations = None

    def forward(self, x):
        if len(x.shape) > 2:
            bs = x.shape[0]
            x = x.view(bs, -1)
        self.activations = x.detach().clone()
        return super(LinearWithActivation, self).forward(x)


class Conv2dWithActivation(nn.Conv2d):
    def __init__(self, module, num_channel=None):
        in_channels = module.in_channels if num_channel is None else num_channel
        super(Conv2dWithActivation, self).__init__(
            in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            padding_mode=module.padding_mode,
            bias=module.bias is not None,
        )
        self.activations = None

    def forward(self, x):
        self.activations = x.detach().clone()
        return super().forward(x)


class BatchNorm2dWithActivation(nn.BatchNorm2d):
    def __init__(self, module):
        super(BatchNorm2dWithActivation, self).__init__(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        self.activations = None

    def forward(self, x):
        self.activations = x.detach().clone()
        return super().forward(x)


class ReLUWithActivation(nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(ReLUWithActivation, self).__init__(*args, **kwargs)
        self.activations = None

    def forward(self, x):
        self.activations = x.detach().clone()
        return super(ReLUWithActivation, self).forward(x)


class MaxPool2dWithActivation(nn.MaxPool2d):
    def __init__(self, module):
        super(MaxPool2dWithActivation, self).__init__(
            module.kernel_size, module.stride, module.padding, module.dilation
        )
        self.activations = None

    def forward(self, x):
        self.activations = x.detach().clone()
        return super(MaxPool2dWithActivation, self).forward(x)


class AdaptiveAvgPool2dWithActivation(nn.AdaptiveAvgPool2d):
    def __init__(self, module: nn.AdaptiveAvgPool2d):
        super(AdaptiveAvgPool2dWithActivation, self).__init__(module.output_size)
        self.activations = None

    def forward(self, x):
        self.activations = x.detach().clone()
        return super(AdaptiveAvgPool2dWithActivation, self).forward(x)


def copy_weights(target_module, source_module):
    if isinstance(
        target_module, (nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.ReLU)
    ) and isinstance(source_module, (nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.ReLU)):
        # Do nothing for layers without weights
        return
    if isinstance(target_module, nn.Linear) and isinstance(source_module, nn.Linear):
        target_module.weight.data.copy_(source_module.weight.data)
        target_module.bias.data.copy_(source_module.bias.data)
    elif isinstance(target_module, nn.Conv2d) and isinstance(source_module, nn.Conv2d):
        target_module.weight.data.copy_(source_module.weight.data)
        if source_module.bias is not None:
            target_module.bias = source_module.bias
    elif isinstance(target_module, nn.BatchNorm2d) and isinstance(
        source_module, nn.BatchNorm2d
    ):
        target_module.weight.data.copy_(source_module.weight.data)
        target_module.bias.data.copy_(source_module.bias.data)
        target_module.running_mean.data.copy_(source_module.running_mean.data)
        target_module.running_var.data.copy_(source_module.running_var.data)
    else:
        raise ValueError(
            f"Unsupported module types for copy_weights source: {source_module} and target: {target_module}"
        )


def replace_modules(model, wrapper):
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            setattr(model, name, replace_modules(module, wrapper))
        else:
            wrapped_module = wrapper(module)
            wrapped_module = copy_weights(module, wrapped_module)
            setattr(model, name, wrapped_module)
    return model


class BasicBlockWithActivation(OriginalBasicBlock):
    def __init__(self, block: OriginalBasicBlock):
        inplanes = block.conv1.in_channels
        planes = block.conv1.out_channels

        super(BasicBlockWithActivation, self).__init__(
            inplanes=inplanes,
            planes=planes,
            stride=block.stride,
            downsample=block.downsample,
        )

    def forward(self, x):
        self.activations = x.detach().clone()
        return super().forward(x)


class BottleneckWithActivation(OriginalBottleneck):
    def __init__(self, block: OriginalBottleneck):
        inplanes = block.conv1.in_channels
        planes = block.conv3.out_channels // block.expansion
        groups = block.conv2.groups
        dilation = block.conv2.dilation
        width = block.conv1.out_channels
        base_width = 64 * width // (groups * planes)

        super(BottleneckWithActivation, self).__init__(
            inplanes=inplanes,
            planes=planes,
            stride=block.stride,
            downsample=block.downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
        )

    def forward(self, x):
        self.activations = x.detach().clone()
        return super().forward(x)


def replace_resnet_modules(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if isinstance(block, OriginalBasicBlock) or isinstance(
                    block, ABNBasicBlock
                ):
                    module[i] = BasicBlockWithActivation(block)
                elif isinstance(block, OriginalBottleneck) or isinstance(
                    block, ABNBottleneck
                ):
                    module[i] = BottleneckWithActivation(block)
            setattr(model, name, module)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            setattr(model, name, AdaptiveAvgPool2dWithActivation(module))
    return model
