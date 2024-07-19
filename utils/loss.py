from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.attention_branch import AttentionBranchModel


def criterion_with_cast_targets(
    criterion: nn.modules.loss._Loss, preds: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the loss after changing the type

    Args:
        criterion(Loss): loss function
        preds(Tensor)  : prediction
        targets(Tensor): label

    Returns:
        torch.Tensor: loss value

    Note:
        The type required by the loss function is different, so we convert it
    """
    if isinstance(criterion, nn.CrossEntropyLoss):
        # targets = F.one_hot(targets, num_classes=2)
        targets = targets.long()

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        targets = F.one_hot(targets, num_classes=2)
        targets = targets.to(preds.dtype)

    return criterion(preds, targets)


def calculate_loss(
    criterion: nn.modules.loss._Loss,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    lambdas: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Calculate the loss
    Add the attention loss when AttentionBranchModel

    Args:
        criterion(Loss)  : Loss function
        preds(Tensor)    : Prediction
        targets(Tensor)  : Label
        model(nn.Module) : Model that made the prediction
        lambdas(Dict[str, float]): Weight of each term of the loss

    Returns:
        torch.Tensor: Loss value
    """
    loss = criterion_with_cast_targets(criterion, outputs, targets)

    # Attention Loss
    if isinstance(model, AttentionBranchModel):
        keys = ["att", "var"]
        if lambdas is None:
            lambdas = {key: 1 for key in keys}
        for key in keys:
            if key not in lambdas:
                lambdas[key] = 1

        attention_loss = criterion_with_cast_targets(
            criterion, model.attention_pred, targets
        )
        # loss = loss + attention_loss
        # attention = model.attention_branch.attention
        # _, _, W, H = attention.size()

        # att_sum = torch.sum(attention, dim=(-1, -2))
        # attention_loss = torch.mean(att_sum / (W * H))
        loss = loss + lambdas["att"] * attention_loss
        # attention = model.attention_branch.attention
        # attention_varmean = attention.var(dim=(1, 2, 3)).mean()
        # loss = loss - lambdas["var"] * attention_varmean

    return loss
