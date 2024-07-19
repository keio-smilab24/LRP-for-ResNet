import math
import os
from typing import Dict, Union

import cv2
import numpy as np
import skimage.measure
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import get_parameter_depend_in_data_set
from metrics.base import Metric
from utils.utils import reverse_normalize
from utils.visualize import save_data_as_plot, save_image


class PatchInsertionDeletion(Metric):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        patch_size: int,
        step: int,
        dataset: str,
        device: torch.device,
    ) -> None:
        self.total = 0
        self.total_insertion = 0
        self.total_deletion = 0
        self.class_insertion: Dict[int, float] = {}
        self.num_by_classes: Dict[int, int] = {}
        self.class_deletion: Dict[int, float] = {}

        self.model = model
        self.batch_size = batch_size
        self.step = step
        self.device = device
        self.patch_size = patch_size
        self.dataset = dataset

    def evaluate(
        self,
        image: np.ndarray,
        attention: np.ndarray,
        label: Union[np.ndarray, torch.Tensor],
    ) -> None:
        self.image = image.copy()
        self.label = int(label.item())

        # image (C, W, H), attention (1, W', H') -> attention (W, H)
        self.attention = attention
        if not (self.image.shape[1:] == attention.shape):
            self.attention = cv2.resize(
                attention[0], dsize=(self.image.shape[1], self.image.shape[2])
            )

        # Divide attention map into patches and calculate the order of patches
        self.divide_attention_map_into_patch()
        self.calculate_attention_order()

        # Create input for insertion and inference
        self.generate_insdel_images(mode="insertion")
        self.ins_preds = self.inference()  # for plot
        self.ins_auc = auc(self.ins_preds)
        self.total_insertion += self.ins_auc
        del self.input

        # deletion
        self.generate_insdel_images(mode="deletion")
        self.del_preds = self.inference()
        self.del_auc = auc(self.del_preds)
        self.total_deletion += self.del_auc
        del self.input

        self.total += 1

    def divide_attention_map_into_patch(self):
        assert self.attention is not None

        self.patch_attention = skimage.measure.block_reduce(
            self.attention, (self.patch_size, self.patch_size), np.max
        )

    def calculate_attention_order(self):
        attention_flat = np.ravel(self.patch_attention)
        # Sort in descending order
        order = np.argsort(-attention_flat)

        W, H = self.attention.shape
        patch_w, _ = W // self.patch_size, H // self.patch_size
        self.order = np.apply_along_axis(
            lambda x: map_2d_indices(x, patch_w), axis=0, arr=order
        )

    def generate_insdel_images(self, mode: str):
        C, W, H = self.image.shape
        patch_w, patch_h = W // self.patch_size, H // self.patch_size
        num_insertion = math.ceil(patch_w * patch_h / self.step)

        params = get_parameter_depend_in_data_set(self.dataset)
        self.input = np.zeros((num_insertion, C, W, H))
        mean, std = params["mean"], params["std"]
        image = reverse_normalize(self.image.copy(), mean, std)

        for i in range(num_insertion):
            step_index = min(self.step * (i + 1), self.order.shape[1] - 1)
            w_indices = self.order[0, step_index]
            h_indices = self.order[1, step_index]
            threthold = self.patch_attention[w_indices, h_indices]

            if mode == "insertion":
                mask = np.where(threthold <= self.patch_attention, 1, 0)
            elif mode == "deletion":
                mask = np.where(threthold <= self.patch_attention, 0, 1)

            mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_NEAREST)

            for c in range(min(3, C)):
                self.input[i, c] = (image[c] * mask - mean[c]) / std[c]

    def inference(self):
        inputs = torch.Tensor(self.input)

        num_iter = math.ceil(inputs.size(0) / self.batch_size)
        result = torch.zeros(0)

        for iter in range(num_iter):
            start = self.batch_size * iter
            batch_inputs = inputs[start : start + self.batch_size].to(self.device)

            outputs = self.model(batch_inputs)

            outputs = F.softmax(outputs, 1)
            outputs = outputs[:, self.label]
            result = torch.cat([result, outputs.cpu().detach()], dim=0)

        return np.nan_to_num(result)

    def save_images(self):
        params = get_parameter_depend_in_data_set(self.dataset)
        for i, image in enumerate(self.input):
            save_image(image, f"tmp/{self.total}_{i}", params["mean"], params["std"])

    def score(self) -> Dict[str, float]:
        result = {
            "Insertion": self.insertion(),
            "Deletion": self.deletion(),
            "PID": self.insertion() - self.deletion(),
        }

        for class_idx in self.class_insertion.keys():
            self.class_insertion_score(class_idx)
            self.class_deletion_score(class_idx)

        return result

    def log(self) -> str:
        result = "Class\tPID\tIns\tDel\n"

        scores = self.score()
        result += f"All\t{scores['PID']:.3f}\t{scores['Insertion']:.3f}\t{scores['Deletion']:.3f}\n"

        for class_idx in self.class_insertion.keys():
            pid = scores[f"PID_{class_idx}"]
            insertion = scores[f"Insertion_{class_idx}"]
            deletion = scores[f"Deletion_{class_idx}"]
            result += f"{class_idx}\t{pid:.3f}\t{insertion:.3f}\t{deletion:.3f}\n"

        return result

    def insertion(self) -> float:
        return self.total_insertion / self.total

    def deletion(self) -> float:
        return self.total_deletion / self.total

    def class_insertion_score(self, class_idx: int) -> float:
        num_samples = self.num_by_classes[class_idx]
        inserton_score = self.class_insertion[class_idx]

        return inserton_score / num_samples

    def class_deletion_score(self, class_idx: int) -> float:
        num_samples = self.num_by_classes[class_idx]
        deletion_score = self.class_deletion[class_idx]

        return deletion_score / num_samples

    def clear(self) -> None:
        self.total = 0
        self.ins_preds = None
        self.del_preds = None

    def save_roc_curve(self, save_dir: str) -> None:
        ins_fname = os.path.join(save_dir, f"{self.total}_insertion.png")
        save_data_as_plot(self.ins_preds, ins_fname, label=f"AUC = {self.ins_auc:.4f}")

        del_fname = os.path.join(save_dir, f"{self.total}_deletion.png")
        save_data_as_plot(self.del_preds, del_fname, label=f"AUC = {self.del_auc:.4f}")


def map_2d_indices(indices_1d: int, width: int):
    """
    Convert 1D index to 2D index
    1D index is converted to 2D index

    Args:
        indices_1d(array): index
        width(int)       : width

    Examples:
        [[0, 1, 2], [3, 4, 5]]
        -> [0, 1, 2, 3, 4, 5]

        map_2d_indices(1, 3)
        >>> [0, 1]
        map_ed_indices(5, 3)
        >>> [1, 2]

        Return the index of the array before flattening
    """
    return [indices_1d // width, indices_1d % width]


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
