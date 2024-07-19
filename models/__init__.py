import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import (
    ResNet,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.models.vgg import VGG

from models.attention_branch import add_attention_branch
from models.lrp import replace_resnet_modules

ALL_MODELS = [
    "resnet18",
    "resnet34",
    "resnet",
    "resnet50",
    "resnet50-legacy",
    "EfficientNet",
    "wide",
    "vgg",
    "vgg19",
    "vgg19_bn",
    "vgg19_skip",
    "swin",
]


class OneWayResNet(nn.Module):
    def __init__(self, model: ResNet) -> None:
        super().__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = model.avgpool
        self.classifier = nn.Sequential(model.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _vgg19_skip_forward:
    def __init__(self, model_ref: VGG) -> None:
        self.model_ref = model_ref

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
        x = self.model_ref.features[0:2](x)  # conv 3->64, relu
        x = self.model_ref.features[2:4](x) + x  # conv 64->64, relu
        x = self.model_ref.features[4:5](x)  # maxpool
        x = self.model_ref.features[5:7](x)  # conv 64->128, relu
        x = self.model_ref.features[7:9](x) + x  # conv 128->128, relu
        x = self.model_ref.features[9:10](x)  # maxpool
        x = self.model_ref.features[10:12](x)  # conv 128->256, relu
        x = self.model_ref.features[12:14](x) + x  # conv 256->256, relu
        x = self.model_ref.features[14:16](x) + x  # conv 256->256, relu
        x = self.model_ref.features[16:18](x) + x  # conv 256->256, relu
        x = self.model_ref.features[18:19](x)  # maxpool
        x = self.model_ref.features[19:21](x)  # conv 256->512, relu
        x = self.model_ref.features[21:23](x) + x  # conv 512->512, relu
        x = self.model_ref.features[23:25](x) + x  # conv 512->512, relu
        x = self.model_ref.features[25:27](x) + x  # conv 512->512, relu
        x = self.model_ref.features[27:28](x)  # maxpool
        x = self.model_ref.features[28:30](x) + x  # conv 512->512, relu
        x = self.model_ref.features[30:32](x) + x  # conv 512->512, relu
        x = self.model_ref.features[32:34](x) + x  # conv 512->512, relu
        x = self.model_ref.features[34:36](x) + x  # conv 512->512, relu
        x = self.model_ref.features[36:37](x)  # maxpool
        # x = self.model_ref.features(x)
        x = self.model_ref.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model_ref.classifier(x)
        return x


def create_model(
    base_model: str,
    num_classes: int = 1000,
    num_channel: int = 3,
    base_pretrained: Optional[str] = None,
    base_pretrained2: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    attention_branch: bool = False,
    division_layer: Optional[str] = None,
    theta_attention: float = 0,
    init_classifier: bool = True,
) -> Optional[nn.Module]:
    """
    Create model from model name and parameters

    Args:
        base_model(str)       : Base model name (resnet, etc.)
        num_classes(int)      : Number of classes
        base_pretrained(str)  : Base model pretrain path
        base_pretrained2(str) : Pretrain path after changing the final layer
        pretrained_path(str)  : Pretrain path of the final model
                                (After attention branch, etc.)
        attention_branch(bool): Whether to attention branch
        division_layer(str)   : Which layer to introduce attention branch
        theta_attention(float): Threshold when entering Attention Branch
                                Set pixels with lower attention than this value to 0 and input

    Returns:
        nn.Module: Created model
    """
    # Create base model
    if base_model == "resnet":
        model = resnet18(pretrained=(base_pretrained is not None))
        layer_index = {"layer1": -6, "layer2": -5, "layer3": -4}
        if init_classifier:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model == "resnet18":
        model = resnet18(pretrained=True)
        if init_classifier:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model == "resnet34":
        model = resnet34(pretrained=True)
        if init_classifier:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model == "resnet50":
        model = resnet50(pretrained=True)
        if init_classifier:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model == "resnet50-v2":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if init_classifier:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif base_model == "vgg":
        model = torchvision.models.vgg11(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif base_model == "vgg19":
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif base_model == "vgg19_bn":
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif base_model == "vgg19_skip":
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        model.forward = _vgg19_skip_forward(model)
    else:
        return None

    # Load if base_pretrained is path
    if base_pretrained is not None and os.path.isfile(base_pretrained):
        model.load_state_dict(torch.load(base_pretrained))
        print(f"base pretrained {base_pretrained} loaded.")

    if attention_branch:
        assert division_layer is not None
        model = add_attention_branch(
            model,
            layer_index[division_layer],
            num_classes,
            theta_attention,
        )
        model.attention_branch = replace_resnet_modules(model.attention_branch)

    # Load if pretrained is path
    if pretrained_path is not None and os.path.isfile(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"pretrained {pretrained_path} loaded.")

    return model
