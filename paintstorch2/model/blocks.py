from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ε = 1e-8


class ModConv2D(nn.Module):
    def __init__(
        self,
        ichannels: int,
        ochannels: int,
        *,
        kernel_size: int,
        demod: bool = True,
    ) -> None:
        super(ModConv2D, self).__init__()
        self.ichannels = ichannels
        self.ochannels = ochannels
        self.kernel_size = kernel_size
        self.demod = demod
        
        weight_size = ochannels, ichannels, kernel_size, kernel_size
        self.weight = nn.Parameter(torch.randn(*weight_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="leaky_relu")

    @property
    def padding(self) -> int:
        return (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        wy = y[:, None, :, None, None]
        wo = self.weight[None, :, :, :, :]
        weights = wo * (wy + 1)

        if self.demod:
            dim = 2, 3, 4
            d = torch.rsqrt((weights ** 2).sum(dim=dim, keepdims=True) + ε)
            weights = weights * d

        x = x.view(1, -1, h, w)
        _, _, *size = weights.size()
        weights = weights.view(-1, *size)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.view(-1, self.ochannels, h, w)


class ToRGB(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        ichannels: int,
        outchannels: int = 3,
        upsample: bool = False,
    ) -> None:
        super(ToRGB, self).__init__()
        self.style_mapping = nn.Linear(latent_dim, ichannels)
        self.mod_conv = ModConv2D(
            ichannels, outchannels, kernel_size=1, demod=False,
        )
        
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        ) if upsample else None

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, style: torch.Tensor,
    ) -> torch.Tensor:
        x = self.mod_conv(x, self.style_mapping(style))
        
        if residual is not None:
            x = x + residual
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        ichannels: int,
        ochannels: int,
        upsample: bool = True,
        upsample_rgb: bool = True,
    ) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        ) if upsample else None

        self.style1 = nn.Linear(latent_dim, ichannels)
        self.noise1 = nn.Conv2d(1, ochannels, kernel_size=1)
        self.mode_conv1 = ModConv2D(ichannels, ochannels, kernel_size=3)

        self.style2 = nn.Linear(latent_dim, ochannels)
        self.noise2 = nn.Conv2d(1, ochannels, kernel_size=1)
        self.mode_conv2 = ModConv2D(ochannels, ochannels, kernel_size=3)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = ToRGB(latent_dim, ochannels, upsample=upsample_rgb)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        if self.upsample is not None:
            x = self.upsample(x)
        
        b, c, h, w = x.size()
        noise = noise[:, :, :h, :w]
        noise1 = self.noise1(noise)
        noise2 = self.noise2(noise)

        style1 = self.style1(style)
        style2 = self.style2(style)

        x = self.activation(self.mode_conv1(x, style1) + noise1)
        x = self.activation(self.mode_conv2(x, style2) + noise2)

        rgb = self.to_rgb(x, residual, style)
        return x, rgb


class ResNetXtBootleneck(nn.Module):
    def __init__(
        self,
        ichannels: int,
        ochannels: int,
        cardinality: int,
        stride: int = 1,
        dilate: int = 1,
    ) -> None:
        super(ResNetXtBootleneck, self).__init__()
        hchannels = ochannels // 2
        
        self.reduce = nn.Conv2d(
            ichannels, hchannels, kernel_size=1, stride=1, bias=False,
        )

        self.conv = nn.Conv2d(
            hchannels, hchannels,
            kernel_size=2 + stride,
            stride=stride,
            padding=dilate,
            groups=cardinality,
            bias=False,
        )
        
        self.expand = nn.Conv2d(
            hchannels, ochannels, kernel_size=1, stride=1, bias=False,
        )
        
        self.shortcut = nn.AvgPool2d(2, stride=stride) if stride > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.reduce(x)
        bottleneck = self.conv(bottleneck)
        bottleneck = self.expand(bottleneck)

        if self.shortcut is not None:
            bottleneck = bottleneck + self.shortcut(x)
        
        return bottleneck


if __name__ == "__main__":
    x = torch.rand((2, 3, 64, 64))
    y = torch.rand((2, 3))
    z = torch.rand((2, 32))
    n = torch.rand((2, 1, 64, 64))

    out = ModConv2D(3, 8, kernel_size=3)(x, y)
    out = ToRGB(32, 8, upsample=True)(out, x, z)

    out, rgb = UpsampleBlock(32, 3, 8, upsample=False)(x, x, z, n)
    out = DownsampleBlock(8, 1, downsample=True)(out)