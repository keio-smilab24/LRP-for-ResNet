import argparse
import os
from typing import Any, Dict, Optional, Tuple, Union

import captum
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.transform import resize
from torchinfo import summary
from torchvision.models.resnet import ResNet
from tqdm import tqdm

import wandb
from data import ALL_DATASETS, create_dataloader_dict, get_parameter_depend_in_data_set
from metrics.base import Metric
from metrics.patch_insdel import PatchInsertionDeletion
from models import ALL_MODELS, OneWayResNet, create_model
from models.attention_branch import AttentionBranchModel
from models.lrp import *
from models.rise import RISE
from src.lrp import abn_lrp, basic_lrp, resnet_lrp
from src.utils import SkipConnectionPropType
from utils.utils import fix_seed, parse_with_config
from utils.visualize import (
    save_image,
    save_image_with_attention_map,
    simple_plot_and_save,
)


def calculate_attention(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    method: str,
    rise_params: Optional[Dict[str, Any]],
    fname: Optional[str],
    skip_connection_prop_type: SkipConnectionPropType = "latest",
) -> Tuple[np.ndarray, Optional[float]]:
    relevance_out = None
    if method.lower() == "abn":
        assert isinstance(model, AttentionBranchModel)
        model(image)
        attentions = model.attention_branch.attention  # (1, W, W)
        attention = attentions[0]
        attention: np.ndarray = attention[0].detach().cpu().numpy()
    elif method.lower() == "rise":
        assert rise_params is not None
        rise_model = RISE(
            model,
            n_masks=rise_params["n_masks"],
            p1=rise_params["p1"],
            input_size=rise_params["input_size"],
            initial_mask_size=rise_params["initial_mask_size"],
            n_batch=rise_params["n_batch"],
            mask_path=rise_params["mask_path"],
        )
        attentions = rise_model(image)  # (N_class, W, H)
        attention = attentions[label]
        attention: np.ndarray = attention.cpu().numpy()
    elif method.lower() == "npy":
        assert fname is not None
        attention: np.ndarray = np.load(fname)
    elif method.lower() == "gradcam":
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = [ClassifierOutputTarget(pred_label)]
        grayscale_cam = cam(input_tensor=image, targets=targets)
        attention = grayscale_cam[0, :]
    elif method.lower() == "scorecam":
        from src import scorecam as scam
        resnet_model_dict = dict(type='resnet50', arch=model, layer_name='layer4', input_size=(224, 224))
        cam = scam.ScoreCAM(resnet_model_dict)
        attention = cam(image)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() == "lrp":  # ours
        if isinstance(model, ResNet):
            model = OneWayResNet(model)
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention: np.ndarray = basic_lrp(
            model, image, rel_pass_ratio=1.0, topk=1, skip_connection_prop=skip_connection_prop_type
        ).detach().cpu().numpy()
    elif method.lower() == "captumlrp":
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.LRP(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() == "captumlrp-positive":
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.LRP(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy().clip(0, None)
    elif method.lower() in ["captumgradxinput", "gradxinput"]:
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.InputXGradient(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() in ["captumig", "ig"]:
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.IntegratedGradients(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() in ["captumig-positive", "ig-positive"]:
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.IntegratedGradients(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy().clip(0, None)
    elif method.lower() in ["captumguidedbackprop", "guidedbackprop", "gbp"]:
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.GuidedBackprop(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() in ["captumguidedbackprop-positive", "guidedbackprop-positive", "gbp-positive"]:
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.GuidedBackprop(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy().clip(0, None)
    # elif method.lower() in ["smoothgrad", "vargrad"]:  # OOM...
    #     tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
    #     pred_label = tpl.indices[0].item()
    #     relevance_out = tpl.values[0].item()
    #     ig = captum.attr.IntegratedGradients(model)
    #     attention = captum.attr.NoiseTunnel(ig).attribute(image, nt_type=method.lower(), nt_samples=5, target=pred_label)
    #     attention = attention[0].detach().cpu().numpy()
    elif method.lower() == "lime":
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.Lime(model).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    elif method.lower() == "captumgradcam":
        tpl = torch.softmax(model(image), dim=-1).max(dim=-1)
        pred_label = tpl.indices[0].item()
        relevance_out = tpl.values[0].item()
        attention = captum.attr.LayerGradCam(model, model.layer4[-1]).attribute(image, target=pred_label)
        attention = attention[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Invalid method was requested: {method}")

    return attention, relevance_out


def apply_gaussian_to_max_pixel(mask, kernel_size=5, sigma=1):
    # Find max value's indices
    max_val_indices = np.where(mask == np.amax(mask))

    # Initialize a new mask with the same size of the original mask
    new_mask = np.zeros(mask.shape)

    # Assign the max value of the original mask to the corresponding location of the new mask
    new_mask[max_val_indices] = mask[max_val_indices]

    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(new_mask, (kernel_size, kernel_size), sigma)

    return blurred_mask


def remove_other_components(mask, threshold=0.5):
    # Binarize the mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Detect connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # Calculate the maximum value in the original mask for each component
    max_values = [np.max(mask[labels == i]) for i in range(num_labels)]

    # Find the component with the largest maximum value
    largest_max_value_component = np.argmax(max_values)
    # second_index = np.argsort(max_values)[-2]
    # third_index = np.argsort(max_values)[-3]
    # print(max_values)
    # print(largest_max_value_component)
    # print(second_index)

    # Create a new mask where all components other than the one with the largest max value are removed
    first_mask = np.where(labels == largest_max_value_component, mask*3, 0)
    # second_mask = np.where(labels == second_index, mask, 0)
    # third_mask = np.where(labels == third_index, mask / 3, 0)
    new_mask = first_mask
    # new_mask = first_mask + second_mask + third_mask

    return new_mask


def apply_heat_quantization(attention, q_level: int = 8):
    max_ = attention.max()
    min_ = attention.min()

    # quantization
    bin = np.linspace(min_, max_, q_level)

    # apply quantization
    for i in range(q_level - 1):
        attention[(attention >= bin[i]) & (attention < bin[i + 1])] = bin[i]

    return attention


# @torch.no_grad()
def visualize(
    dataloader: data.DataLoader,
    model: nn.Module,
    method: str,
    batch_size: int,
    patch_size: int,
    step: int,
    save_dir: str,
    all_class: bool,
    params: Dict[str, Any],
    device: torch.device,
    evaluate: bool = False,
    attention_dir: Optional[str] = None,
    use_c1c: bool = False,
    heat_quantization: bool = False,
    hq_level: int = 8,
    skip_connection_prop_type: SkipConnectionPropType = "latest",
    data_limit: int = -1,
) -> Union[None, Metric]:
    if evaluate:
        metrics = PatchInsertionDeletion(
            model, batch_size, patch_size, step, params["name"], device
        )
        insdel_save_dir = os.path.join(save_dir, "insdel")
        if not os.path.isdir(insdel_save_dir):
            os.makedirs(insdel_save_dir)

    rel_ins = []
    rel_outs = []

    counter = {}

    model.eval()
    # # print accuracy on test set
    # total = 0
    # correct = 0
    # for i, data_ in enumerate(
    #     tqdm(dataloader, desc="Count failures", dynamic_ncols=True)
    # ):
    #     inputs, labels = (
    #         data_[0].to(device),
    #         data_[1].to(device),
    #     )
    #     outputs = model(inputs)
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    # print(f"Total: {total}")
    # print(f"Correct: {correct}")
    # print(f"Accuracy: {correct / total:.4f}")

    # Inference time estimation
    elapsed_time_sum = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    batch = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(100):
        start_event.record()
        model.forward(batch)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_sum += start_event.elapsed_time(end_event)
    print(f"Average inference time: {elapsed_time_sum / 100}")

    # Elapsed time to inference + generate explanation
    elapsed_time_sum = 0.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    image = torch.randn(1, 3, 224, 224).to(device)
    if isinstance(model, ResNet):
        oneway_model = OneWayResNet(model)
    for _ in range(100):
        start_event.record()
        basic_lrp(
            oneway_model, image, rel_pass_ratio=1.0, topk=1, skip_connection_prop=skip_connection_prop_type
        )
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_sum += start_event.elapsed_time(end_event)
    print(f"Average time to inference and generate an attribution map: {elapsed_time_sum / 100}")

    for i, data_ in enumerate(
        tqdm(dataloader, desc="Visualizing: ", dynamic_ncols=True)
    ):
        if data_limit > 0 and i >= data_limit:
            break
        torch.cuda.memory_summary(device=device)
        inputs, labels = (
            data_[0].to(device),
            data_[1].to(device),
        )
        image: torch.Tensor = inputs[0].cpu().numpy()
        label: torch.Tensor = labels[0]

        if label != 1 and not all_class:
            continue

        counter.setdefault(label.item(), 0)
        counter[label.item()] += 1
        n_eval_per_class = 1000 / len(params["classes"])
        # n_eval_per_class = 5  # TODO: try
        if counter[label.item()] > n_eval_per_class:
            continue

        base_fname = f"{i+1}_{params['classes'][label]}"

        attention_fname = None
        if attention_dir is not None:
            attention_fname = os.path.join(attention_dir, f"{base_fname}.npy")

        attention, rel_ = calculate_attention(
            model, inputs, label, method, params, attention_fname, skip_connection_prop_type
        )
        if rel_ is not None:
            rel_ins.append(attention.sum().item())
            rel_outs.append(rel_)
        if use_c1c:
            attention = resize(attention, (28, 28))
            attention = remove_other_components(attention, threshold=attention.mean())

        if heat_quantization:
            attention = apply_heat_quantization(attention, hq_level)

        if attention is None:
            continue
        if method == "RISE":
            np.save(f"{save_dir}/{base_fname}.npy", attention)

        if evaluate:
            metrics.evaluate(
                image.copy(),
                attention,
                label,
            )
            metrics.save_roc_curve(insdel_save_dir)
            base_fname = f"{base_fname}_{metrics.ins_auc - metrics.del_auc:.4f}"
            if i % 50 == 0:
                print(metrics.log())

        save_fname = os.path.join(save_dir, f"{base_fname}.png")
        save_image_with_attention_map(
            image, attention, save_fname, params["mean"], params["std"]
        )

        save_image(image, save_fname[:-4]+".original.png", params["mean"], params["std"])

    # Plot conservations
    conservation_dir = os.path.join(save_dir, "conservation")
    if not os.path.isdir(conservation_dir):
        os.makedirs(conservation_dir)
    simple_plot_and_save(rel_ins, rel_outs, os.path.join(conservation_dir, "plot.png"))

    if evaluate:
        return metrics


def main(args: argparse.Namespace) -> None:
    fix_seed(args.seed, args.no_deterministic)

    # データセットの作成
    dataloader_dict = create_dataloader_dict(
        args.dataset, 1, args.image_size, only_test=True, shuffle_val=True, dataloader_seed=args.seed
    )
    dataloader = dataloader_dict["Test"]
    assert isinstance(dataloader, data.DataLoader)

    params = get_parameter_depend_in_data_set(args.dataset)

    mask_path = os.path.join(args.root_dir, "masks.npy")
    if not os.path.isfile(mask_path):
        mask_path = None
    rise_params = {
        "n_masks": args.num_masks,
        "p1": args.p1,
        "input_size": (args.image_size, args.image_size),
        "initial_mask_size": (args.rise_scale, args.rise_scale),
        "n_batch": args.batch_size,
        "mask_path": mask_path,
    }
    params.update(rise_params)

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
        init_classifier=args.dataset != "ImageNet",  # Use pretrained classifier in ImageNet
    )
    assert model is not None, "Model name is invalid"

    # run_nameをpretrained pathから取得
    # checkpoints/run_name/checkpoint.pt -> run_name
    run_name = args.pretrained.split(os.sep)[-2] if args.pretrained is not None else "pretrained"
    save_dir = os.path.join(
        "outputs",
        f"{run_name}_{args.notes}_{args.method}{args.block_size}",
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    summary(
        model,
        (args.batch_size, params["num_channel"], args.image_size, args.image_size),
    )

    model.to(device)

    wandb.init(project=args.dataset, name=run_name, notes=args.notes)
    wandb.config.update(vars(args))

    metrics = visualize(
        dataloader,
        model,
        args.method,
        args.batch_size,
        args.block_size,
        args.insdel_step,
        save_dir,
        args.all_class,
        params,
        device,
        args.visualize_only,
        attention_dir=args.attention_dir,
        use_c1c=args.use_c1c,
        heat_quantization=args.heat_quantization,
        hq_level=args.hq_level,
        skip_connection_prop_type=args.skip_connection_prop_type,
        data_limit=args.data_limit,
    )

    if hasattr(args, "test_acc"):
        print(f"Test Acc: {args.test_acc}")

    if metrics is not None:
        print(metrics.log())
        for key, value in metrics.score().items():
            wandb.run.summary[key] = value


def parse_args() -> argparse.Namespace:
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
        "--orig_model",
        action="store_true",
        help="calc insdel score by using original model",
    )
    parser.add_argument(
        "--theta_att", type=float, default=0, help="threthold of attention branch"
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
    parser.add_argument("--data_limit", type=int, default=-1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="*",
        default=[1.0, 1.0],
        help="weights for label by class",
    )

    parser.add_argument("--root_dir", type=str, default="./outputs/")
    parser.add_argument("--visualize_only", action="store_false")
    parser.add_argument("--all_class", action="store_true")
    # recommend (step, size) in 512x512 = (1, 10000), (2, 2500), (4, 500), (8, 100), (16, 20), (32, 10), (64, 5), (128, 1)
    # recommend (step, size) in 224x224 = (1, 500), (2, 100), (4, 20), (8, 10), (16, 5), (32, 1)
    parser.add_argument("--insdel_step", type=int, default=500)
    parser.add_argument("--block_size", type=int, default=1)

    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--sign", type=str, default="all")

    parser.add_argument("--num_masks", type=int, default=5000)
    parser.add_argument("--rise_scale", type=int, default=9)
    parser.add_argument(
        "--p1", type=float, default=0.3, help="percentage of mask [pixel = (0, 0, 0)]"
    )

    parser.add_argument("--attention_dir", type=str, help="path to attention npy file")

    parser.add_argument("--use-c1c", action="store_true", help="use C1C technique")

    parser.add_argument("--heat-quantization", action="store_true", help="use heat quantization technique")
    parser.add_argument("--hq-level", type=int, default=8, help="number of quantization level")

    parser.add_argument("--skip-connection-prop-type", type=str, default="latest", help="type of skip connection propagation")

    return parse_with_config(parser)


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(parse_args())
