import argparse
import json
import random
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn


def fix_seed(seed: int, deterministic: bool = False) -> None:
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def reverse_normalize(
    x: np.ndarray,
    mean: Union[Tuple[float], Tuple[float, float, float]],
    std: Union[Tuple[float], Tuple[float, float, float]],
):
    """
    Restore normalization

    Args:
        x(ndarray) : Matrix that has been normalized
        mean(Tuple): Mean specified at the time of normalization
        std(Tuple) : Standard deviation specified at the time of normalization
    """
    if x.shape[0] == 1:
        x = x * std + mean
        return x
    x[0, :, :] = x[0, :, :] * std[0] + mean[0]
    x[1, :, :] = x[1, :, :] * std[1] + mean[1]
    x[2, :, :] = x[2, :, :] * std[2] + mean[2]

    return x


def module_generator(model: nn.Module, reverse: bool = False):
    """
    Generator for nested Module, can handle nested Sequential in one layer
    Note that you cannot get layers by index

    Args:
        model (nn.Module): Model
        reverse(bool)    : Whether to reverse

    Yields:
        Each layer of the model
    """
    modules = list(model.children())
    if reverse:
        modules = modules[::-1]

    for module in modules:
        if list(module.children()):
            yield from module_generator(module, reverse)
            continue
        yield module


def save_json(data: Union[List, Dict], save_path: str) -> None:
    """
    Save list/dict to json

    Args:
        data (List/Dict): Data to save
        save_path(str)  : Path to save (including extension)
    """
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)


def softmax_image(image: torch.Tensor) -> torch.Tensor:
    image_size = image.size()
    if len(image_size) == 4:
        B, C, W, H = image_size
    elif len(image_size) == 3:
        B = 1
        C, W, H = image_size
    else:
        raise ValueError

    image = image.view(B, C, W * H)
    image = torch.softmax(image, dim=-1)

    image = image.view(B, C, W, H)
    if len(image_size) == 3:
        image = image[0]

    return image


def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        result: np.ndarray = tensor.cpu().detach().numpy()
    else:
        result = tensor

    return result


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Coexistence of argparse and json file config

    Args:
        parser(ArgumentParser)

    Returns:
        Namespace: Can be used in the same way as argparser

    Note:
        Values specified in the arguments take precedence over the config
    """
    args, _unknown = parser.parse_known_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args
