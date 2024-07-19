import warnings
from typing import Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from utils.utils import reverse_normalize, tensor_to_numpy


def save_attention_map(attention: Union[np.ndarray, torch.Tensor], fname: str) -> None:
    attention = tensor_to_numpy(attention)

    fig, ax = plt.subplots()

    min_att = attention.min()
    max_att = attention.max()

    im = ax.imshow(
        attention, interpolation="nearest", cmap="jet", vmin=min_att, vmax=max_att
    )
    fig.colorbar(im)
    plt.savefig(fname)
    plt.clf()
    plt.close()


def save_image_with_attention_map(
    image: np.ndarray,
    attention: np.ndarray,
    fname: str,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    only_img: bool = False,
    normalize: bool = False,
    sign: Literal["all", "positive", "negative", "absolute_value"] = "all",
) -> None:
    if len(attention.shape) == 3:
        attention = attention[0]

    image = image[:3]
    mean = mean[:3]
    std = std[:3]

    # attention = (attention - attention.min()) / (attention.max() - attention.min())
    if normalize:
        attention = normalize_attr(attention, sign)

    # image : (C, W, H)
    attention = cv2.resize(attention, dsize=(image.shape[1], image.shape[2]))
    image = reverse_normalize(image.copy(), mean, std)
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots()
    if only_img:
        ax.axis('off')  # No axes for a cleaner look
    if image.shape[2] == 1:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(image, vmin=0, vmax=1)

    im = ax.imshow(attention, cmap="jet", alpha=0.4, vmin=attention.min(), vmax=attention.max())
    if not only_img:
        fig.colorbar(im)

    if only_img:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(fname)

    plt.clf()
    plt.close()


def save_image(image: np.ndarray, fname: str, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
    # image : (C, W, H)
    image = reverse_normalize(image.copy(), mean, std)
    image = image.clip(0, 1)
    image = np.transpose(image, (1, 2, 0))

    # Convert image from [0, 1] float to [0, 255] uint8 for saving
    image = (image * 255).astype(np.uint8)
    # image = image.astype(np.uint8)

    if image.shape[2] == 1:  # if grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(fname, image)


def simple_plot_and_save(x, y, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=x, mode='lines', line=dict(color='#999999', width=2)))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='#82B366', size=12)))
    fig.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        # xaxis_title=r"$\Huge {\sum_i R^{(\textrm{First block})}_i}$",
        # yaxis_title=r"$\Huge {p(\hat y_c)}$",
        xaxis=dict(
            title_standoff=28,
            tickfont=dict(family="Times New Roman", size=30, color="black"),
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickcolor='black',
        ),
        yaxis=dict(
            title_standoff=26,
            tickfont=dict(family="Times New Roman", size=30, color="black"),
            linecolor='black',
            showgrid=False,
            ticks='outside',
            tickcolor='black',
            scaleanchor="x",
            scaleratio=1,
        ),
        autosize=False,
        width=600,
        height=600,
        # margin=dict(l=115, r=5, b=115, t=5),
        margin=dict(l=5, r=5, b=5, t=5),
    )
    fig.write_image(save_path)


def simple_plot_and_save_legacy(x, y, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color='orange')
    plt.plot(x, x, color='#999999')  # y=x line
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(save_path)


def save_data_as_plot(
    data: np.ndarray,
    fname: str,
    x: Optional[np.ndarray] = None,
    label: Optional[str] = None,
    xlim: Optional[Union[int, float]] = None,
) -> None:
    """
    Save data as plot

    Args:
        data(ndarray): Data to save
        fname(str)   : File name to save
        x(ndarray)   : X-axis
        label(str)   : Label of legend
    """
    fig, ax = plt.subplots()

    if x is None:
        x = range(len(data))

    ax.plot(x, data, label=label)

    xmax = len(data) if xlim is None else xlim
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.05, 1.05)

    plt.legend()
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05)
    plt.clf()
    plt.close()


"""followings are borrowed & modified from captum.attr._utils"""

def _normalize_scale(attr: np.ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values: np.ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def normalize_attr(
    attr: np.ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)

    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if sign == "all":
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif sign == "positive":
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif sign == "negative":
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif sign == "absolute_value":
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)
