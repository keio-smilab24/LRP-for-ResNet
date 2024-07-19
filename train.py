import argparse
import datetime
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import wandb
from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from evaluate import test
from metrics.base import Metric
from models import ALL_MODELS, create_model
from models.attention_branch import AttentionBranchModel
from models.lrp import BottleneckWithActivation
from optim import ALL_OPTIM, create_optimizer
from optim.sam import SAM
from utils.loss import calculate_loss
from utils.utils import fix_seed, module_generator, parse_with_config, save_json
from utils.visualize import save_attention_map


class EarlyStopping:
    """
    Attributes:
        patience(int): How long to wait after last time validation loss improved.
        delta(float) : Minimum change in the monitored quantity to qualify as an improvement.
        save_dir(str): Directory to save a model when improvement is found.
    """

    def __init__(
        self, patience: int = 7, delta: float = 0, save_dir: str = "."
    ) -> None:
        self.patience = patience
        self.delta = delta
        self.save_dir = save_dir

        self.counter: int = 0
        self.early_stop: bool = False
        self.best_val_loss: float = np.Inf

    def __call__(self, val_loss: float, net: nn.Module) -> str:
        if val_loss + self.delta < self.best_val_loss:
            log = f"({self.best_val_loss:.5f} --> {val_loss:.5f})"
            self._save_checkpoint(net)
            self.best_val_loss = val_loss
            self.counter = 0
            return log

        self.counter += 1
        log = f"(> {self.best_val_loss:.5f} {self.counter}/{self.patience})"
        if self.counter >= self.patience:
            self.early_stop = True
        return log

    def _save_checkpoint(self, net: nn.Module) -> None:
        save_path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save(net.state_dict(), save_path)


def set_parameter_trainable(module: nn.Module, is_trainable: bool = True) -> None:
    """
    Set all parameters of the module to is_trainable(bool)

    Args:
        module(nn.Module): Target module
        is_trainable(bool): Whether to train the parameters
    """
    for param in module.parameters():
        param.requires_grad = is_trainable


def set_trainable_bottlenecks(model, num_trainable):
    # 2. Set the last num_trainable Bottleneck/BottleneckWithActivation layers as trainable
    count = 0
    for child in reversed(list(model.children())):
        # Go through each layer in the current sequential block
        for layer in reversed(list(child.children())):
            if isinstance(layer, (BottleneckWithActivation)):
                for param in layer.parameters():
                    param.requires_grad = True
                count += 1
                if count == num_trainable:
                    return


def freeze_model(
    model: nn.Module,
    num_trainable_module: int = 0,
    fe_trainable: bool = False,
    ab_trainable: bool = False,
    perception_trainable: bool = False,
    final_trainable: bool = True,
) -> None:
    """
    Freeze the model
    After freezing the whole model, only the final layer is trainable
    Then, the last num_trainable_module are trainable from the back

    Args:
        num_trainable_module(int): Number of trainable modules
        fe_trainable(bool): Whether to train the Feature Extractor
        ab_trainable(bool): Whether to train the Attention Branch
        perception_trainable(bool): Whether to train the Perception Branch

    Note:
        (fe|ab|perception)_trainable is only used when AttentionBranchModel
        num_trainable_module takes precedence over the above    
    """
    if isinstance(model, AttentionBranchModel):
        set_parameter_trainable(model.feature_extractor, fe_trainable)
        set_parameter_trainable(model.attention_branch, ab_trainable)
        set_parameter_trainable(model.perception_branch, perception_trainable)
        modules = module_generator(model.perception_branch, reverse=True)
    else:
        if num_trainable_module < 0:
            set_parameter_trainable(model)
            return
        set_parameter_trainable(model, is_trainable=False)
        modules = module_generator(model, reverse=True)

    final_layer = modules.__next__()
    set_parameter_trainable(final_layer, final_trainable)
    # set_parameter_trainable(model.perception_branch[0], False)
    # set_parameter_trainable(model.feature_extractor[0], True)
    # set_parameter_trainable(model.feature_extractor[1], True)

    # for i, module in enumerate(modules):
    #     if num_trainable_module <= i:
    #         break

        # set_parameter_trainable(module)
    set_trainable_bottlenecks(model, num_trainable_module)


def setting_learning_rate(
    model: nn.Module, lr: float, lr_linear: float, lr_ab: Optional[float] = None
) -> Iterable:
    """
    Set learning rate for each layer

    Args:
        model (nn.Module): Model to set learning rate
        lr(float)       : Learning rate for the last layer/Attention Branch
        lr_linear(float): Learning rate for the last layer
        lr_ab(float)    : Learning rate for Attention Branch

    Returns:
        Iterable with learning rate
        It is given to the argument of optim.Optimizer
    """
    if isinstance(model, AttentionBranchModel):
        if lr_ab is None:
            lr_ab = lr_linear
        params = [
            {"params": model.attention_branch.parameters(), "lr": lr_ab},
            {"params": model.perception_branch[:-1].parameters(), "lr": lr},
            {"params": model.perception_branch[-1].parameters(), "lr": lr_linear},
        ]
    else:
        try:
            params = [
                {"params": model[:-1].parameters(), "lr": lr},
                {"params": model[-1].parameters(), "lr": lr_linear},
            ]
        except TypeError:
            params = [{"params": model.parameters(), "lr": lr}]

    return params


def wandb_log(loss: float, metrics: Metric, phase: str) -> None:
    """
    Output logs to wandb
    Add phase to each metric for easy understanding
    (e.g. Acc -> Train_Acc)

    Args:
        loss(float)    : Loss value
        metircs(Metric): Evaluation metrics
        phase(str)     : train / val / test    
    """
    log_items = {f"{phase}_loss": loss}

    for metric, value in metrics.score().items():
        log_items[f"{phase}_{metric}"] = value

    wandb.log(log_items)


def train_insdel(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.modules.loss._Loss,
    mode: str,
    theta_dist: List[float] = [0.3, 0.5, 0.7],
):
    assert isinstance(model, AttentionBranchModel)
    attention_map = model.attention_branch.attention
    attention_map = F.interpolate(attention_map, images.shape[2:])
    att_base = attention_map.max()

    theta = random.choice(theta_dist)
    assert mode in ["insertion", "deletion"]
    if mode == "insertion":
        labels = torch.ones_like(labels)
        attention_map = torch.where(attention_map > att_base * theta, 1.0, 0.0)
    if mode == "deletion":
        labels = torch.zeros_like(labels)
        attention_map = torch.where(attention_map > att_base * theta, 0.0, 1.0)

    inputs = images * attention_map
    output = model(inputs.float())
    loss = criterion(output, labels)

    return loss


def train(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    metric: Metric,
    lambdas: Optional[Dict[str, float]] = None,
    saliency: bool = False,
) -> Tuple[float, Metric]:
    total = 0
    total_loss: float = 0
    torch.autograd.set_detect_anomaly(True)

    model.train()
    for data_ in tqdm(dataloader, desc="Train: ", dynamic_ncols=True):
        inputs, labels = (
            data_[0].to(device),
            data_[1].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = calculate_loss(criterion, outputs, labels, model, lambdas)
        loss.backward()
        total_loss += loss.item()

        metric.evaluate(outputs, labels)

        # When the optimizer is SAM, backward twice
        if isinstance(optimizer, SAM):
            optimizer.first_step(zero_grad=True)
            loss_sam = calculate_loss(criterion, model(inputs), labels, model, lambdas)
            loss_sam.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()

        total += labels.size(0)

    train_loss = total_loss / total
    return train_loss, metric


def main(args: argparse.Namespace):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    fix_seed(args.seed, args.no_deterministic)

    # Create dataloaders
    dataloader_dict = create_dataloader_dict(
        args.dataset,
        args.batch_size,
        args.image_size,
        train_ratio=args.train_ratio,
    )
    data_params = get_parameter_depend_in_data_set(
        args.dataset, pos_weight=torch.Tensor(args.loss_weights).to(device)
    )

    # Create a model
    model = create_model(
        args.model,
        num_classes=len(data_params["classes"]),
        num_channel=data_params["num_channel"],
        base_pretrained=args.base_pretrained,
        base_pretrained2=args.base_pretrained2,
        pretrained_path=args.pretrained,
        attention_branch=args.add_attention_branch,
        division_layer=args.div,
        theta_attention=args.theta_att,
    )
    assert model is not None, "Model name is invalid"
    freeze_model(
        model,
        args.trainable_module,
        "fe" not in args.freeze,
        "ab" not in args.freeze,
        "pb" not in args.freeze,
        "linear" not in args.freeze,
    )

    # Setup optimizer and scheduler
    params = setting_learning_rate(model, args.lr, args.lr_linear, args.lr_ab)
    optimizer = create_optimizer(
        args.optimizer, params, args.lr, args.weight_decay, args.momentum
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=args.min_lr, warmup_t=10, warmup_prefix=True)

    if args.saliency_guided:
        set_parameter_trainable(model.perception_branch[0], False)

    criterion = data_params["criterion"]
    metric = data_params["metric"]

    # Create run_name (for save_dir / wandb)
    if args.model is not None:
        config_file = os.path.basename(args.config)
        run_name = os.path.splitext(config_file)[0]
    else:
        run_name = args.model
    run_name += ["", f"_div{args.div}"][args.add_attention_branch]
    run_name = f"{run_name}_{now_str}"
    if args.run_name is not None:
        run_name = args.run_name

    save_dir = os.path.join(args.save_dir, run_name)
    assert not os.path.isdir(save_dir)
    os.makedirs(save_dir)
    best_path = os.path.join(save_dir, "best.pt")

    configs = vars(args)
    configs.pop("config")  # To prevent the old config from being included in the new config applied with the model

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience, save_dir=save_dir
    )

    wandb.init(project=args.dataset, name=run_name, notes=args.notes)
    wandb.config.update(configs)
    configs["pretrained"] = best_path
    save_json(configs, os.path.join(save_dir, "config.json"))

    # Model details display (torchsummary)
    summary(
        model,
        (args.batch_size, data_params["num_channel"], args.image_size, args.image_size),
    )

    lambdas = {"att": args.lambda_att}

    save_test_acc = 0
    model.to(device)
    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}]")
        for phase, dataloader in dataloader_dict.items():
            if phase == "Train":
                loss, metric = train(
                    dataloader,
                    model,
                    criterion,
                    optimizer,
                    metric,
                    lambdas=lambdas,
                )
            else:
                loss, metric = test(
                    dataloader,
                    model,
                    criterion,
                    metric,
                    device,
                    phase,
                    lambdas=lambdas,
                )

            metric_log = metric.log()
            log = f"{phase}\t| {metric_log} Loss: {loss:.5f} "

            wandb_log(loss, metric, phase)

            if phase == "Val":
                early_stopping_log = early_stopping(loss, model)
                log += early_stopping_log
                scheduler.step(loss)

            print(log)
            if phase == "Test" and not early_stopping.early_stop:
                save_test_acc = metric.acc()

            metric.clear()
            if args.add_attention_branch:
                save_attention_map(
                    model.attention_branch.attention[0][0], "attention.png"
                )

        if early_stopping.early_stop:
            print("Early Stopping")
            model.load_state_dict(torch.load(os.path.join(save_dir, "checkpoint.pt")))
            break

    torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
    configs["test_acc"] = save_test_acc.item()
    save_json(configs, os.path.join(save_dir, "config.json"))
    wandb.log({"final_test_acc": save_test_acc})
    print("Training Finished")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, help="path to config file (json)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_deterministic", action="store_false")

    parser.add_argument("-n", "--notes", type=str, default="")

    # Model
    parser.add_argument("-m", "--model", choices=ALL_MODELS, help="model name")
    parser.add_argument(
        "-add_ab",
        "--add_attention_branch",
        action="store_true",
        help="add Attention Branch",
    )
    parser.add_argument(
        "--div",
        type=str,
        choices=["layer1", "layer2", "layer3"],
        default="layer2",
        help="place to attention branch",
    )
    parser.add_argument("--base_pretrained", type=str, help="path to base pretrained")
    parser.add_argument(
        "--base_pretrained2",
        type=str,
        help="path to base pretrained2 ( after change_num_classes() )",
    )
    parser.add_argument("--pretrained", type=str, help="path to pretrained")
    parser.add_argument(
        "--theta_att", type=float, default=0, help="threthold of attention branch"
    )

    # Freeze
    parser.add_argument(
        "--freeze",
        type=str,
        nargs="*",
        choices=["fe", "ab", "pb", "linear"],
        default=[],
        help="freezing layer",
    )
    parser.add_argument(
        "--trainable_module",
        type=int,
        default=-1,
        help="number of trainable modules, -1: all trainable",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="ratio for train val split"
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="weights for label by class",
    )

    # Optimizer
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "-optim", "--optimizer", type=str, default="AdamW", choices=ALL_OPTIM
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr_linear",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--lr_ab",
        "--lr_attention_branch",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--factor", type=float, default=0.3333, help="new_lr = lr * factor"
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=2,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )

    parser.add_argument(
        "--lambda_att", type=float, default=0.1, help="weights for attention loss"
    )

    parser.add_argument(
        "--early_stopping_patience", type=int, default=6, help="Early Stopping patience"
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints", help="path to save checkpoints"
    )

    parser.add_argument(
        "--run_name", type=str, help="save in save_dir/run_name and wandb name"
    )

    return parse_with_config(parser)


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
