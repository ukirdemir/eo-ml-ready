"""A small U-Net that predicts the cold-cloud mask at t+60 from four input channels."""

from __future__ import annotations

import torch
from torch import nn


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Three-level U-Net; output is one logit per pixel."""

    def __init__(self, in_channels: int = 4, base: int = 16) -> None:
        super().__init__()
        self.enc1 = double_conv(in_channels, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.bottom = double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = double_conv(base * 8 + base * 4, base * 4)
        self.dec2 = double_conv(base * 4 + base * 2, base * 2)
        self.dec1 = double_conv(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with the same height and width as x."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottom = self.bottom(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(bottom), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
