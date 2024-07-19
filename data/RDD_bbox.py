import csv
import os
from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class RDDBboxDataset(Dataset):

    def __init__(
        self, root: str, image_set: str = "train", transform: Optional[Callable] = None
    ) -> None:
        super().__init__()

        self.root = root
        self.image_set = image_set
        self.transform = transform

        self.base_dir = os.path.join(self.root, "RDD_bbox")
        annot_dir = "Annotations"
        annotation_file = f"{image_set}.csv"
        self.annotation_file = os.path.join(self.base_dir, annot_dir, annotation_file)

        with open(self.annotation_file) as f:
            reader = csv.reader(f)
            # skip header
            reader.__next__()
            annotations = [row for row in reader]

        # Transpose [Image_fname, Label]
        annotations = [list(x) for x in zip(*annotations)]

        self.images = list(
            map(lambda x: os.path.join(self.base_dir, x), annotations[0])
        )
        self.targets = list(map(int, annotations[1]))

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self) -> int:
        return len(self.images)
