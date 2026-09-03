"""PyTorch ResNet-BS bone-suppression model, adapted from the validated
port at analyses/bone_suppression_sample/pytorch_port/model.py (35 Conv2D
layers, same residual topology and 0.1 residual scaling as the vendored
Keras original at external/CXR-bone-suppression/bone_suppression.ipynb,
no BatchNorm, verified there against the real Keras predictions). Weights
at weights/resnet-bonesuppression-jsrt/resnet_bs_weights.npz.

Fully convolutional, same spatial resolution in and out: the Keras
original always resizes to 256x256 first, but the architecture has no
resolution-dependent layer, and the port's own run_sample.py already
validated it running directly at 512x512 with no degradation. This
project's own use feeds it the already center-cropped, already-square
512x512 preprocessed image directly, no extra resize, so bone suppression
adds no geometry of its own: the output aligns pixel-for-pixel with the
input, and no additional label transform is needed beyond whatever the
512x512 preprocessing step itself already requires.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, filters: int, scaling: float | None):
        super().__init__()
        self.conv_a = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv_b = nn.Conv2d(filters, filters, 3, padding=1)
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = torch.relu(self.conv_a(x))
        b = self.conv_b(b)
        if self.scaling:
            b = b * self.scaling
        return x + b


class ResNetBS(nn.Module):
    def __init__(self, num_filters: int = 64, num_res_blocks: int = 16, res_block_scaling: float | None = 0.1):
        super().__init__()
        self.conv_in = nn.Conv2d(1, num_filters, 3, padding=1)
        self.res_blocks = nn.ModuleList(ResBlock(num_filters, res_block_scaling) for _ in range(num_res_blocks))
        self.conv_mid = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv_out = nn.Conv2d(num_filters, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)
        b = x_in
        for block in self.res_blocks:
            b = block(b)
        b = self.conv_mid(b)
        x = x_in + b
        return self.conv_out(x)

    def conv_layers_in_export_order(self) -> list[nn.Conv2d]:
        layers = [self.conv_in]
        for block in self.res_blocks:
            layers.append(block.conv_a)
            layers.append(block.conv_b)
        layers.append(self.conv_mid)
        layers.append(self.conv_out)
        return layers


def load_ported_weights(model: ResNetBS, npz_path: Path) -> None:
    weights = np.load(npz_path)
    conv_layers = model.conv_layers_in_export_order()
    assert len(conv_layers) == 35, f"expected 35 conv layers, got {len(conv_layers)}"
    with torch.no_grad():
        for i, layer in enumerate(conv_layers):
            w = torch.from_numpy(weights[f"conv{i}_weight"]).float()
            b = torch.from_numpy(weights[f"conv{i}_bias"]).float()
            assert layer.weight.shape == w.shape, f"conv{i}: layer expects {layer.weight.shape}, npz has {w.shape}"
            layer.weight.copy_(w)
            layer.bias.copy_(b)
    model.eval()


def load_model(weights_npz: Path, device: torch.device) -> ResNetBS:
    model = ResNetBS()
    load_ported_weights(model, weights_npz)
    return model.to(device)
