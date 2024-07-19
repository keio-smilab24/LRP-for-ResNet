from typing import Iterable

import torch.optim as optim


from optim.sam import SAM

ALL_OPTIM = ["SGD", "Adam", "AdamW", "SAM"]


def create_optimizer(
    optim_name: str,
    params: Iterable,
    lr: float,
    weight_decay: float = 0.9,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Create an optimizer

    Args:
        optim_name(str)    : Name of the optimizer
        params(Iterable)   : params
        lr(float)          : Learning rate
        weight_decay(float): weight_decay
        momentum(float)    : momentum
        
    Returns:
        Optimizer
    """
    assert optim_name in ALL_OPTIM

    if optim_name == "SGD":
        optimizer = optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optim_name == "Adam":
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "AdamW":
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "SAM":
        base_optimizer = optim.SGD
        optimizer = SAM(
            params, base_optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError

    return optimizer
