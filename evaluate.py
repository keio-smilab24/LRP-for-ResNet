import argparse
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from torchinfo import summary
from tqdm import tqdm

from data import (ALL_DATASETS, create_dataloader_dict,
                  get_parameter_depend_in_data_set)
from metrics.base import Metric
from models import ALL_MODELS, create_model
from utils.loss import calculate_loss
from utils.utils import fix_seed, parse_with_config


@torch.no_grad()
def test(
    dataloader: data.DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    metrics: Metric,
    device: torch.device,
    phase: str = "Test",
    lambdas: Optional[Dict[str, float]] = None,
) -> Tuple[float, Metric]:
    total = 0
    total_loss: float = 0

    model.eval()
    for data in tqdm(dataloader, desc=f"{phase}: ", dynamic_ncols=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)

        total_loss += calculate_loss(criterion, outputs, labels, model, lambdas).item()
        metrics.evaluate(outputs, labels)
        total += labels.size(0)

    test_loss = total_loss / total
    return test_loss, metrics


import torchvision.transforms as transforms
from PIL import Image

from utils.visualize import save_image


@torch.no_grad()
def test_with_save(
    dataloader: data.DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    metrics: Metric,
    device: torch.device,
    phase: str = "Test",
    lambdas: Optional[Dict[str, float]] = None,
) -> Tuple[float, Metric]:
    total = 0
    total_loss: float = 0
    transform = transforms.ToPILImage()

    model.eval()
    for data in tqdm(dataloader, desc=f"{phase}: ", dynamic_ncols=True):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        outputs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
        misclassified = (pred != labels).nonzero().squeeze()
        if misclassified.numel() == 1:
            misclassified = [misclassified.item()]

        for index in misclassified:
            image = inputs[index, :3].cpu()
            # image = transform(image)

            true_label = labels[index].item()
            # predicted_label = pred[index].item()
            params = get_parameter_depend_in_data_set("RDD")
            save_image(
                image.detach().cpu().numpy(),
                f"outputs/error_anal/{true_label}/{outputs[index, true_label]:.4f}.png",
                params["mean"],
                params["std"],
            )

            # image.save(
            #     f"outputs/error_anal/{true_label}/{outputs[index, true_label]}.png"
            # )

        total_loss += calculate_loss(criterion, outputs, labels, model, lambdas).item()
        metrics.evaluate(outputs, labels)
        total += labels.size(0)

    test_loss = total_loss / total
    return test_loss, metrics


def main(args: argparse.Namespace) -> None:
    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        args.dataset, args.batch_size, args.image_size, only_test=True
    )
    dataloader = dataloader_dict["Test"]
    assert isinstance(dataloader, data.DataLoader)

    params = get_parameter_depend_in_data_set(args.dataset)

    # モデルの作成
    model = create_model(
        args.model,
        num_classes=len(params["classes"]),
        num_channel=params["num_channel"],
        base_pretrained=args.base_pretrained,
        base_pretrained2=args.base_pretrained2,
        pretrained_path=args.pretrained,
        attention_branch=args.add_attention_branch,
        division_layer=args.div,
        theta_attention=args.theta_att,
    )
    assert model is not None, "Model name is invalid"

    model.load_state_dict(torch.load(args.pretrained))
    print(f"pretrained {args.pretrained} loaded.")

    criterion = params["criterion"]
    metric = params["metric"]

    # run_nameをpretrained pathから取得
    # checkpoints/run_name/checkpoint.pt -> run_name
    run_name = args.pretrained.split(os.sep)[-2]
    save_dir = os.path.join("outputs", run_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    summary(
        model,
        (args.batch_size, params["num_channel"], args.image_size, args.image_size),
    )

    model.to(device)

    lambdas = {"att": args.lambda_att, "var": args.lambda_var}
    loss, metrics = test_with_save(
        dataloader, model, criterion, metric, device, lambdas=lambdas
    )
    metric_log = metrics.log()
    print(f"Test\t| {metric_log} Loss: {loss:.5f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, help="path to config file (json)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false")

    # Model
    parser.add_argument("-m", "--model", choices=ALL_MODELS, help="model name")
    parser.add_argument(
        "-add_ab",
        "--add_attention_branch",
        action="store_true",
        help="add Attention Branch",
    )
    parser.add_argument(
        "-div",
        "--division_layer",
        type=str,
        choices=["layer1", "layer2", "layer3"],
        default="layer2",
        help="place to attention branch",
    )
    parser.add_argument("--base_pretrained", type=str, help="path to base pretrained")
    parser.add_argument("--pretrained", type=str, help="path to pretrained")

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="weights for label by class",
    )

    parser.add_argument(
        "--lambda_att", type=float, default=0.1, help="weights for attention loss"
    )
    parser.add_argument(
        "--lambda_var", type=float, default=0, help="weights for variance loss"
    )

    parser.add_argument("--root_dir", type=str, default="./outputs/")

    return parse_with_config(parser)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
