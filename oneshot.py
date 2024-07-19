""" Visualize attention map of a model with a given image.

Example:
```sh
poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "scorecam" \
    --image-path ./qual/original/painted_bunting.png --label 15 --save-path ./qual/scorecam/painted_bunting.png

poetry run python oneshot.py -c checkpoints/CUB_resnet50_Seed42/config.json --method "lrp" \
    --skip-connection-prop-type "flows_skip" --heat-quantization \
    --image-path ./qual/original/painted_bunting.png --label 15 --save-path ./qual/ours/painted_bunting.png

poetry run python oneshot.py -c configs/ImageNet_resnet50.json --method "lrp" \
    --skip-connection-prop-type "flows_skip" --heat-quantization \
    --image-path ./qual/original/bee.png --label 309 --save-path ./qual/ours/bee.png
```

"""

import argparse
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize

from data import ALL_DATASETS, get_parameter_depend_in_data_set
from metrics.base import Metric
from models import ALL_MODELS, create_model
from models.lrp import *
from src.utils import SkipConnectionPropType
from utils.utils import fix_seed, parse_with_config
from utils.visualize import (
    save_image_with_attention_map,
)
from visualize import (  # TODO: move these functions to src directory
    apply_heat_quantization,
    calculate_attention,
    remove_other_components,
)


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image


def visualize(
    image_path: str,
    label: int,
    model: nn.Module,
    method: str,
    save_path: str,
    params: Dict[str, Any],
    device: torch.device,
    attention_dir: Optional[str] = None,
    use_c1c: bool = False,
    heat_quantization: bool = False,
    hq_level: int = 8,
    skip_connection_prop_type: SkipConnectionPropType = "latest",
    normalize: bool = False,
    sign: str = "all",
) -> Union[None, Metric]:
    model.eval()
    torch.cuda.memory_summary(device=device)
    inputs = load_image(image_path, params["input_size"][0]).unsqueeze(0).to(device)
    image = inputs[0].cpu().numpy()
    label = torch.tensor(label).to(device)

    attention, _ = calculate_attention(
        model, inputs, label, method, params, None, skip_connection_prop_type
    )
    if use_c1c:
        attention = resize(attention, (28, 28))
        attention = remove_other_components(attention, threshold=attention.mean())

    if heat_quantization:
        attention = apply_heat_quantization(attention, hq_level)

    save_path = save_path + ".png" if save_path[-4:] != ".png" else save_path
    save_image_with_attention_map(
        image, attention, save_path, params["mean"], params["std"], only_img=True, normalize=normalize, sign=sign
    )


def main(args: argparse.Namespace) -> None:
    fix_seed(args.seed, args.no_deterministic)
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

    # Create model
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

    model.to(device)
    visualize(
        args.image_path,
        args.label,
        model,
        args.method,
        args.save_path,
        params,
        device,
        attention_dir=args.attention_dir,
        use_c1c=args.use_c1c,
        heat_quantization=args.heat_quantization,
        hq_level=args.hq_level,
        skip_connection_prop_type=args.skip_connection_prop_type,
        normalize=args.normalize,
        sign=args.sign,
    )


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

    # Target Image
    parser.add_argument("--image-path", type=str, help="path to target image")
    parser.add_argument("--label", type=int, help="label of target image")

    # Visualize option
    parser.add_argument("--normalize", action="store_true", help="normalize attribution")
    parser.add_argument("--sign", type=str, default="all", help="sign of attribution to show")

    # Save
    parser.add_argument("--save-path", type=str, help="path to save image")

    # Dataset
    parser.add_argument("--dataset", type=str, default="IDRiD", choices=ALL_DATASETS)
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
