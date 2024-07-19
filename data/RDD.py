import csv
import os
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RDDDataset(Dataset):

    def __init__(
        self, root: str, image_set: str = "train", transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform

        self.base_dir = os.path.join(self.root, "RDD")

        image_dir = "JPEGImages"
        annot_dir = "Annotations"
        sam_dir = "SAM-mask"
        annotation_file = f"{image_set}.csv"

        self.image_dir = os.path.join(self.base_dir, image_dir)
        self.sam_dir = os.path.join(self.base_dir, sam_dir)
        self.annotation_file = os.path.join(self.base_dir, annot_dir, annotation_file)

        with open(self.annotation_file) as f:
            reader = csv.reader(f)
            # skip header
            reader.__next__()
            annotations = [row for row in reader]

        # Transpose [Image_fname, Label]
        annotations = [list(x) for x in zip(*annotations)]

        image_fnames = annotations[0]
        self.images = list(
            map(lambda x: os.path.join(self.image_dir, x + ".jpg"), image_fnames)
        )
        self.sam_segments = list(
            map(lambda x: os.path.join(self.sam_dir, x + ".png"), image_fnames)
        )
        self.targets = self.targets = list(
            map(lambda x: int(1 <= int(x)), annotations[1])
        )

    def __getitem__(self, index) -> Tuple[Any, Any, Any, Any]:
        orig_image = Image.open(self.images[index]).convert("RGB")

        sam = cv2.imread(self.sam_segments[index], cv2.IMREAD_GRAYSCALE)
        sam_orig = np.expand_dims(cv2.resize(sam, (orig_image.size[:2])), -1)
        sam_mask = 1 * (sam_orig > 0)
        if sam_mask.sum() < sam_mask.size * 0.1:
            sam_mask = np.ones_like(sam_mask)

        # Calculate the bounding box coordinates of the mask
        mask_coords = np.argwhere(sam_mask.squeeze())
        min_y, min_x = np.min(mask_coords, axis=0)
        max_y, max_x = np.max(mask_coords, axis=0)

        # Crop the image using the bounding box coordinates
        image = orig_image.crop((min_x, min_y, max_x + 1, max_y + 1))

        # image = np.array(orig_image) * sam_mask
        # image = Image.fromarray(image.astype(np.uint8))

        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)
            orig_image = self.transform(orig_image)

        sam_resized = np.expand_dims(cv2.resize(sam, (orig_image.shape[1:])), -1)
        sam_mask = 1 * (sam_resized > 0)
        sam_mask = sam_mask.reshape(1, orig_image.shape[1], orig_image.shape[2])
        sam_mask = torch.from_numpy(sam_mask.astype(np.uint8)).clone()

        return image, target, sam_mask, orig_image

    def __len__(self) -> int:
        return len(self.images)
