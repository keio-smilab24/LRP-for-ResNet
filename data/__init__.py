import random
from typing import Any, Callable, Dict, Optional

import numpy
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch import nn

from data.CUB import CUB_CLASSES, CUBDataset
from data.imagenet import IMAGENET_CLASSES, ImageNetDataset
from data.RDD import RDDDataset
from data.RDD_bbox import RDDBboxDataset
from data.sampler import BalancedBatchSampler
from metrics.accuracy import Accuracy, MultiClassAccuracy

ALL_DATASETS = ["RDD", "RDD_bbox", "CUB", "ImageNet"]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def create_dataloader_dict(
    dataset_name: str,
    batch_size: int,
    image_size: int = 224,
    only_test: bool = False,
    train_ratio: float = 0.9,
    shuffle_val: bool = False,
    dataloader_seed: int = 0,
) -> Dict[str, data.DataLoader]:
    """
    Create dataloader dictionary

    Args:
        dataset_name(str) : Dataset name
        batch_size  (int) : Batch size
        image_size  (int) : Image size
        only_test   (bool): Create only test dataset
        train_ratio(float): Train / val split ratio when there is no val

    Returns:
        dataloader_dict : Dataloader dictionary
    """

    test_dataset = create_dataset(dataset_name, "test", image_size)
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle_val, worker_init_fn=seed_worker, generator=get_generator(dataloader_seed),
    )

    if only_test:
        return {"Test": test_dataloader}

    train_dataset = create_dataset(
        dataset_name,
        "train",
        image_size,
    )

    dataset_params = get_parameter_depend_in_data_set(dataset_name)

    # Create val or split
    if dataset_params["has_val"]:
        val_dataset = create_dataset(
            dataset_name,
            "val",
            image_size,
        )
    else:
        train_size = int(train_ratio * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = data.random_split(
            train_dataset, [train_size, val_size]
        )
    val_dataset.transform = test_dataset.transform

    if dataset_params["sampler"]:
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_sampler=BalancedBatchSampler(train_dataset, 2, batch_size // 2),
            worker_init_fn=seed_worker,
            generator=get_generator(dataloader_seed),
        )
    else:
        train_dataloader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=get_generator(dataloader_seed)
        )
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=shuffle_val, worker_init_fn=seed_worker, generator=get_generator(dataloader_seed)
    )

    dataloader_dict = {
        "Train": train_dataloader,
        "Val": val_dataloader,
        "Test": test_dataloader,
    }

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)},  Test: {len(test_dataset)}")
    return dataloader_dict


def create_dataset(
    dataset_name: str,
    image_set: str = "train",
    image_size: int = 224,
    transform: Optional[Callable] = None,
) -> data.Dataset:
    """
    Create dataset
    Normalization parameters are created for each dataset

    Args:
        dataset_name(str)  : Dataset name
        image_set(str)     : Choose from train / val / test
        image_size(int)    : Image size
        transform(Callable): transform

    Returns:
        data.Dataset : PyTorch dataset
    """
    assert dataset_name in ALL_DATASETS
    params = get_parameter_depend_in_data_set(dataset_name)

    if transform is None:
        if image_set == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize(int(image_size/0.875)),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(params["mean"], params["std"]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(int(image_size/0.875)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(params["mean"], params["std"]),
                ]
            )

    if params["has_params"]:
        dataset = params["dataset"](
            root="./datasets",
            image_set=image_set,
            params=params,
            transform=transform,
        )
    else:
        dataset = params["dataset"](
            root="./datasets", image_set=image_set, transform=transform
        )

    return dataset


def get_parameter_depend_in_data_set(
    dataset_name: str,
    pos_weight: torch.Tensor = torch.Tensor([1]),
    dataset_root: str = "./datasets",
) -> Dict[str, Any]:
    """
    Get parameters of the dataset
    
    Args:
        dataset_name(str): Dataset name

    Returns:
        dict[str, Any]: Parameters such as mean, variance, class name, etc.
    """
    params = dict()
    params["name"] = dataset_name
    params["root"] = dataset_root
    # Whether to pass params to the dataset class
    params["has_params"] = False
    params["num_channel"] = 3
    params["sampler"] = True
    # ImageNet
    params["mean"] = (0.485, 0.456, 0.406)
    params["std"] = (0.229, 0.224, 0.225)

    if dataset_name == "RDD":
        params["dataset"] = RDDDataset
        params["num_channel"] = 3
        params["mean"] = (0.4770, 0.5026, 0.5094)
        params["std"] = (0.2619, 0.2684, 0.3001)
        params["classes"] = ("no crack", "crack")
        params["has_val"] = False
        params["has_params"] = False
        params["sampler"] = False

        params["metric"] = Accuracy()
        params["criterion"] = nn.CrossEntropyLoss()
    elif dataset_name == "RDD_bbox":
        params["dataset"] = RDDBboxDataset
        params["num_channel"] = 3
        params["mean"] = (0.4401, 0.4347, 0.4137)
        params["std"] = (0.2016, 0.1871, 0.1787)
        params["classes"] = ("no crack", "crack")
        params["has_val"] = False
        params["has_params"] = False
        params["sampler"] = False
        params["metric"] = Accuracy()
        params["criterion"] = nn.CrossEntropyLoss()
    elif dataset_name == "CUB":
        params["dataset"] = CUBDataset
        params["num_channel"] = 3
        # params["mean"] = (0.4859, 0.4996, 0.4318)
        # params["std"] = (0.2266, 0.2218, 0.2609)
        params["mean"] = (0.485, 0.456, 0.406)  # trace ImageNet
        params["std"] = (0.229, 0.224, 0.225)  # trace ImageNet
        params["classes"] = CUB_CLASSES
        params["has_val"] = False
        params["has_params"] = False
        params["sampler"] = False
        params["metric"] = MultiClassAccuracy()
        params["criterion"] = nn.CrossEntropyLoss()
    elif dataset_name == "ImageNet":
        params["dataset"] = ImageNetDataset
        params["num_channel"] = 3
        params["mean"] = (0.485, 0.456, 0.406)
        params["std"] = (0.229, 0.224, 0.225)
        params["classes"] = IMAGENET_CLASSES
        params["has_val"] = False  # Use val. set as the test set
        params["has_params"] = False
        params["sampler"] = False
        params["metric"] = MultiClassAccuracy()
        params["criterion"] = nn.CrossEntropyLoss()

    return params
