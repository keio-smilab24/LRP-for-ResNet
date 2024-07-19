import argparse
from typing import Tuple

import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm

from data import ALL_DATASETS, create_dataset, get_generator, seed_worker
from utils.utils import fix_seed


def calc_mean_std(
    dataloader: data.DataLoader, image_size
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and variance of the image dataset

    Args:
        dataloader(DataLoader): DataLoader
        image_size(int)  : Image size

    Returns:
        Mean and variance of each channel
        Tuple[torch.Tensor, torch.Tensor]    
    """

    sum = torch.tensor([0.0, 0.0, 0.0])
    sum_square = torch.tensor([0.0, 0.0, 0.0])
    total = 0

    for inputs, _ in tqdm(dataloader, dynamic_ncols=True):
        inputs.to(device)
        sum += inputs.sum(axis=[0, 2, 3])
        sum_square += (inputs**2).sum(axis=[0, 2, 3])
        total += inputs.size(0)

    count = total * image_size * image_size

    total_mean = sum / count
    total_var = (sum_square / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std


def main():
    fix_seed(args.seed, True)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )

    dataset = create_dataset(args.dataset, "train", args.image_size, transform)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=get_generator())

    mean, std = calc_mean_std(dataloader, args.image_size)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", default=32, type=int)

    args = parser.parse_args()

    main()
