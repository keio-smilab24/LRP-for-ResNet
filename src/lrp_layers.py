"""Layers for layer-wise relevance propagation.

Layers for layer-wise relevance propagation can be modified.

"""
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from src.lrp_filter import relevance_filter


class RelevancePropagationBasicBlock(nn.Module):
    def __init__(
        self,
        layer: BasicBlock,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
        ]
        self.downsample = layer.downsample
        self.relu = layer.relu
        self.eps = eps
        self.top_k = top_k

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())

    def mainstream_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()  # ignore output of this basic block
        activations = [a.data.requires_grad_(True) for a in activations]

        # NOW, IGNORES DOWN-SAMPLING & SKIP CONNECTION
        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in

    def shortcut_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return r
        a = a.data.requires_grad_(True)
        assert isinstance(self.downsample[0], nn.Conv2d)
        return RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        ratio = self._calc_mainstream_flow_ratio(a)
        assert r.shape == ratio.shape
        r_mainstream = ratio * r
        r_shortcut = (1 - ratio) * r
        r_mainstream = self.mainstream_backward(a, r_mainstream)
        r_shortcut = self.shortcut_backward(a, r_shortcut)
        return r_mainstream + r_shortcut


class RelevancePropagationBasicBlockSimple(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBasicBlockFlowsPureSkip(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())


class RelevancePropagationBasicBlockSimpleFlowsPureSkip(RelevancePropagationBasicBlock):
    """ Relevance propagation for BasicBlock Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: BasicBlock, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.full_like(input_, 0.5)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBottleneck(nn.Module):
    def __init__(
        self,
        layer: Bottleneck,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = [
            layer.conv1,
            layer.bn1,
            layer.relu,
            layer.conv2,
            layer.bn2,
            layer.relu,
            layer.conv3,
            layer.bn3,
        ]
        self.downsample = layer.downsample
        self.relu = layer.relu
        self.eps = eps
        self.top_k = top_k

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())

    def mainstream_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            activations = [a]
            for layer in self.layers:
                activations.append(layer.forward(activations[-1]))

        activations.pop()  # ignore output of this bottleneck block
        activations = [a.data.requires_grad_(True) for a in activations]

        # NOW, IGNORES DOWN-SAMPLING & SKIP CONNECTION
        r_out = r
        for layer in self.layers[::-1]:
            a = activations.pop()
            if self.top_k:
                r_out = relevance_filter(r_out, top_k_percent=self.top_k)

            if isinstance(layer, nn.Conv2d):
                r_in = RelevancePropagationConv2d(layer, eps=self.eps, top_k=self.top_k)(
                    a, r_out
                )
            elif isinstance(layer, nn.BatchNorm2d):
                r_in = RelevancePropagationBatchNorm2d(layer, top_k=self.top_k)(a, r_out)
            elif isinstance(layer, nn.ReLU):
                r_in = RelevancePropagationReLU(layer, top_k=self.top_k)(a, r_out)
            else:
                raise NotImplementedError
            r_out = r_in
        return r_in

    def shortcut_backward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return r
        a = a.data.requires_grad_(True)
        assert isinstance(self.downsample[0], nn.Conv2d)
        return RelevancePropagationConv2d(self.downsample[0], eps=self.eps, top_k=self.top_k)(a, r)

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        ratio = self._calc_mainstream_flow_ratio(a)
        assert r.shape == ratio.shape
        r_mainstream = ratio * r
        r_shortcut = (1 - ratio) * r
        r_mainstream = self.mainstream_backward(a, r_mainstream)
        r_shortcut = self.shortcut_backward(a, r_shortcut)
        return r_mainstream + r_shortcut


class RelevancePropagationBottleneckSimple(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.ones_like(input_)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationBottleneckFlowsPureSkip(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        mainstream = input_
        shortcut = input_
        for layer in self.layers:
            mainstream = layer(mainstream)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        assert mainstream.shape == shortcut.shape
        return mainstream.abs() / (shortcut.abs() + mainstream.abs())


class RelevancePropagationBottleneckSimpleFlowsPureSkip(RelevancePropagationBottleneck):
    """ Relevance propagation for Bottleneck Proto A 
    Divide relevance score for plain shortcuts
    """
    def __init__(self, layer: Bottleneck, eps: float = 1.0e-05, top_k: float = 0.0) -> None:
        super().__init__(layer=layer, eps=eps, top_k=top_k)

    def _calc_mainstream_flow_ratio(self, input_: torch.Tensor) -> torch.Tensor:
        if self.downsample is None:
            return torch.full_like(input_, 0.5)
        return torch.full_like(self.downsample(input_), 0.5)


class RelevancePropagationAdaptiveAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D adaptive average pooling.

    Attributes:
        layer: 2D adaptive average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.AdaptiveAvgPool2d,
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationAvgPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D average pooling.

    Attributes:
        layer: 2D average pooling layer.
        eps: A value added to the denominator for numerical stability.

    """

    def __init__(
        self, layer: torch.nn.AvgPool2d, eps: float = 1.0e-05, top_k: float = 0.0
    ) -> None:
        super().__init__()
        self.layer = layer
        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        return r


class RelevancePropagationMaxPool2d(nn.Module):
    """Layer-wise relevance propagation for 2D max pooling.

    Optionally substitutes max pooling by average pooling layers.

    Attributes:
        layer: 2D max pooling layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.MaxPool2d,
        mode: str = "avg",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        if mode == "avg":
            self.layer = torch.nn.AvgPool2d(kernel_size=(2, 2))
        elif mode == "max":
            self.layer = layer

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        # print(f"maxpool2d {r.min()}, {r.max()}")
        return r


class RelevancePropagationConv2d(nn.Module):
    """Layer-wise relevance propagation for 2D convolution.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: 2D convolutional layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Conv2d,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))

        self.eps = eps
        self.top_k = top_k

    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = (r / z).data
        (z * s).sum().backward()
        c = a.grad
        r = (a * c).data
        # print(f"before norm: {r.sum()}")
        # r = (r - r.min()) / (r.max() - r.min())
        # print(f"after norm: {r.sum()}\n")
        if r.shape != a.shape:
            raise RuntimeError("r.shape != a.shape")
        return r


class RelevancePropagationLinear(nn.Module):
    """Layer-wise relevance propagation for linear transformation.

    Optionally modifies layer weights according to propagation rule. Here z^+-rule

    Attributes:
        layer: linear transformation layer.
        eps: a value added to the denominator for numerical stability.

    """

    def __init__(
        self,
        layer: torch.nn.Linear,
        mode: str = "z_plus",
        eps: float = 1.0e-05,
        top_k: float = 0.0,
    ) -> None:
        super().__init__()

        self.layer = layer

        if mode == "z_plus":
            self.layer.weight = torch.nn.Parameter(self.layer.weight.clamp(min=0.0))
            self.layer.bias = torch.nn.Parameter(torch.zeros_like(self.layer.bias))

        self.eps = eps
        self.top_k = top_k

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.top_k:
            r = relevance_filter(r, top_k_percent=self.top_k)
        z = self.layer.forward(a) + self.eps
        s = r / z
        c = torch.mm(s, self.layer.weight)
        r = (a * c).data
        # print(f"Linear {r.min()}, {r.max()}")
        return r


class RelevancePropagationFlatten(nn.Module):
    """Layer-wise relevance propagation for flatten operation.

    Attributes:
        layer: flatten layer.

    """

    def __init__(self, layer: torch.nn.Flatten, top_k: float = 0.0) -> None:
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r = r.view(size=a.shape)
        return r


class RelevancePropagationReLU(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.ReLU, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationBatchNorm2d(nn.Module):
    """Layer-wise relevance propagation for ReLU activation.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.BatchNorm2d, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationDropout(nn.Module):
    """Layer-wise relevance propagation for dropout layer.

    Passes the relevance scores without modification. Might be of use later.

    """

    def __init__(self, layer: torch.nn.Dropout, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r


class RelevancePropagationIdentity(nn.Module):
    """Identity layer for relevance propagation.

    Passes relevance scores without modifying them.

    """

    def __init__(self, layer: nn.Module, top_k: float = 0.0) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r
