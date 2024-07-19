from typing import Dict, List, Tuple

import torch

from metrics.base import Metric


def num_correct_topk(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)
) -> List[int]:
    """
    Calculate top-k Accuracy

    Args:
        output(Tensor)  : Model output
        target(Tensor)  : Label
        topk(Tuple[int]): How many top ranks should be correct

    Returns:
        List[int] : top-k Accuracy
    """
    maxk = max(topk)

    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # [[False, False, True], [F, F, F], [T, F, F]]
    # -> [0, 0, 1, 0, 0, 0, 1, 0, 0] -> 2
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)
    return result


class Accuracy(Metric):
    def __init__(self) -> None:
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.total += labels.size(0)
        self.correct += num_correct_topk(preds, labels)[0]

        preds = torch.max(preds, dim=-1)[1]
        tp, fp, tn, fn = confusion(preds, labels)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def score(self) -> Dict[str, float]:
        return {
            "Acc": self.acc(),
            "TP": int(self.tp),
            "FP": int(self.fp),
            "FN": int(self.fn),
            "TN": int(self.tn),
        }

    def acc(self) -> float:
        return self.correct / self.total

    def clear(self) -> None:
        self.total = 0
        self.correct = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0


def confusion(output, target) -> Tuple[int, int, int, int]:
    """
    Calculate Confusion Matrix

    Args:
        output(Tensor)  : Model output
        target(Tensor)  : Label

    Returns:
        true_positive(int) : Number of TP
        false_positive(int): Number of FP
        true_negative(int) : Number of TN
        false_negative(int): Number of FN
    """

    # TP: 1/1 = 1, FP: 1/0 -> inf, TN: 0/0 -> nan, FN: 0/1 -> 0
    confusion_vector = output / target

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float("inf")).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return (
        int(true_positives),
        int(false_positives),
        int(true_negatives),
        int(false_negatives),
    )


class MultiClassAccuracy(Metric):
    def __init__(self) -> None:
        self.total = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def evaluate(self, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.total += labels.size(0)
        correct_top1, correct_top5 = num_correct_topk(preds, labels, (1, 5))
        self.top1_correct += correct_top1
        self.top5_correct += correct_top5

    def score(self) -> Dict[str, float]:
        return {
            "Top-1 Acc": self.acc(),
            "Top-5 Acc": self.top5_acc(),
        }

    def acc(self) -> float:
        return self.top1_correct / self.total

    def top5_acc(self) -> float:
        return self.top5_correct / self.total

    def clear(self) -> None:
        self.total = 0
        self.top1_correct = 0
        self.top5_correct = 0