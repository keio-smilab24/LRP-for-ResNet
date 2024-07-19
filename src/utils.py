"""Script with helper function."""
from typing import Literal

from models.lrp import *
from src.lrp_layers import *

SkipConnectionPropType = Literal["simple", "flows_skip", "flows_skip_simple", "latest"]

def layers_lookup(version: SkipConnectionPropType = "latest") -> dict:
    """Lookup table to map network layer to associated LRP operation.

    Returns:
        Dictionary holding class mappings.
    """

    # For the purpose of the ablation study on relevance propagation for skip connections
    if version == "simple":
        return layers_lookup_simple()
    elif version == "flows_skip":
        return layers_lookup_flows_pure_skip()
    elif version == "flows_skip_simple":
        return layers_lookup_simple_flows_pure_skip()
    elif version == "latest":
        return layers_lookup_latest()
    else:
        raise ValueError("Invalid version was specified.")


def layers_lookup_latest() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlock,
        Bottleneck: RelevancePropagationBottleneck,
    }
    return lookup_table


def layers_lookup_simple() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockSimple,
        Bottleneck: RelevancePropagationBottleneckSimple,
    }
    return lookup_table


def layers_lookup_flows_pure_skip() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockFlowsPureSkip,
        Bottleneck: RelevancePropagationBottleneckFlowsPureSkip,
    }
    return lookup_table


def layers_lookup_simple_flows_pure_skip() -> dict:
    lookup_table = {
        torch.nn.modules.linear.Linear: RelevancePropagationLinear,
        torch.nn.modules.conv.Conv2d: RelevancePropagationConv2d,
        torch.nn.modules.activation.ReLU: RelevancePropagationReLU,
        torch.nn.modules.dropout.Dropout: RelevancePropagationDropout,
        torch.nn.modules.flatten.Flatten: RelevancePropagationFlatten,
        torch.nn.modules.pooling.AvgPool2d: RelevancePropagationAvgPool2d,
        torch.nn.modules.pooling.MaxPool2d: RelevancePropagationMaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d: RelevancePropagationAdaptiveAvgPool2d,
        AdaptiveAvgPool2dWithActivation: RelevancePropagationAdaptiveAvgPool2d,
        torch.nn.BatchNorm2d: RelevancePropagationBatchNorm2d,
        BatchNorm2dWithActivation: RelevancePropagationBatchNorm2d,
        BasicBlock: RelevancePropagationBasicBlockSimpleFlowsPureSkip,
        Bottleneck: RelevancePropagationBottleneckSimpleFlowsPureSkip,
    }
    return lookup_table
