from typing import List

import numpy as np
import paintstorch2.model.blocks as pt2_blocks
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(
        self,
        image_size: int,
        latent_dim: int,
        capacity: int = 16,
    ) -> None:
        super(Generator, self).__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.n_layers = int(np.log2(image_size) - 1)

        channels = [capacity * (2 ** 1)] + [
            capacity * (2 ** (i + 1)) for i in range(self.n_layers)
        ][::-1]
        iochannels = list(zip(channels[:-1], channels[1:]))

        pad = (3 - 1) // 2
        self.preprocess = nn.Conv2d(3, channels[0], kernel_size=3, padding=pad)
        
        self.downsample = nn.ModuleList([
            pt2_blocks.DownsampleBlock(
                ichannels,
                ochannels,
                downsample=i > 0,
            ) for i, (ochannels, ichannels) in enumerate(iochannels[::-1])
        ])

        self.affine = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
            for channel, _ in iochannels[::-1]
        ])

        self.upsample = nn.ModuleList([
            pt2_blocks.UpsampleBlock(
                latent_dim,
                ichannels,
                ochannels,
                upsample=i > 0,
                upsample_rgb=i < (self.n_layers - 1),
            ) for i, (ichannels, ochannels) in enumerate(iochannels)
        ])

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.preprocess(x)

        affines: List[torch.Tensor] = []
        for downsample, affine in zip(self.downsample, self.affine):
            x = downsample(x)
            affines.append(affine(x))

        rgb = None
        for i, upsample in enumerate(self.upsample):
            affine = affines[len(affines) - i - 1]
            x, rgb = upsample(x, rgb, style, noise)

        return residual + rgb


class Discriminator(nn.Module):
    def __init__(self, image_size: int, capacity: int = 4) -> None:
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.n_layers = int(np.log2(image_size) - 1) + 3

        channels = [capacity * (2 ** (i + 1)) for i in range(self.n_layers)]
        iochannels = list(zip(channels[:-1], channels[1:]))

        pad = (3 - 1) // 2
        self.preprocess = nn.Conv2d(3, channels[0], kernel_size=3, padding=pad)
        self.downsample = nn.ModuleList([
            pt2_blocks.DownsampleBlock(ichannels, ochannels, downsample=i > 0)
            for i, (ichannels, ochannels) in enumerate(iochannels)
        ])

        self.logits = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward( self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        for downsample in self.downsample:
            x = downsample(x)
        x = self.logits(x).view(x.size(0), -1)
        return x


if __name__ == '__main__':
    x = torch.rand((2, 3, 64, 64))
    z = torch.rand((2, 32))
    n = torch.rand((2, 1, 64, 64))

    out = Generator(64, 32)(x, z, n)
    pred = Discriminator(64)(out)