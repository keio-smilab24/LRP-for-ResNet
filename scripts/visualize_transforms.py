import argparse
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from utils.utils import fix_seed, reverse_normalize


def save_normalized_image(
    image: np.ndarray,
    fname: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    image = reverse_normalize(image.copy(), mean, std)
    image = np.transpose(image, (1, 2, 0))

    fig, ax = plt.subplots()
    ax.imshow(image)

    plt.savefig(fname)
    plt.clf()
    plt.close()


def main():
    fix_seed(args.seed, True)

    dataloader = create_dataloader_dict(args.dataset, 1, args.image_size)
    params = get_parameter_depend_in_data_set(args.dataset)

    save_dir = os.path.join(args.root_dir, "transforms")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for phase, inputs in dataloader.items():
        if not phase == "Train":
            continue

        for i, data in enumerate(inputs):
            image = data[0].cpu().numpy()[0]
            save_fname = os.path.join(save_dir, f"{i}.png")
            save_normalized_image(image, save_fname, params["mean"], params["std"])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--root_dir", type=str, default="./outputs/")

    args = parser.parse_args()

    main()
