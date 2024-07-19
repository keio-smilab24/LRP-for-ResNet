import csv
import os
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
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
        sam_dir = "SAM-all"
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

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")

        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        sam = cv2.imread(self.sam_segments[index], cv2.IMREAD_GRAYSCALE)
        # sam = np.expand_dims(cv2.resize(sam, (image.size[:2])), -1)
        sam = cv2.resize(sam, (image.shape[1:]))
        sam_mask = 1 * (sam > 0)
        sam_mask_pil = Image.fromarray(sam_mask.astype(np.uint8), mode="L")

        mask = torchvision.transforms.functional.to_tensor(sam_mask_pil)
        # print(image.shape, mask.shape)
        image = torch.cat((image, mask), 0)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
