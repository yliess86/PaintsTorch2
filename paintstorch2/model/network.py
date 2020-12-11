from typing import List

import numpy as np
import paintstorch2.model.blocks as pt2_blocks
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.channels = [32, 64, 128, 512]
        
        iochannels = list(zip([
            self.channels[0]] + self.channels[:-1], self.channels,
        ))

        self.preprocess = nn.Conv2d(
            3, self.channels[0], kernel_size=3, padding=(3 - 1) // 2,
        )

        self.downsample = nn.ModuleList([
            pt2_blocks.DownsampleBlock(
                ichannels, ochannels, downsample=i > 0,
            ) for i, (ichannels, ochannels) in enumerate(iochannels)
        ])

        self.affine = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=1, bias=False)
            for channel in self.channels
        ])

        self.upsample = nn.ModuleList([
            pt2_blocks.UpsampleBlock(
                latent_dim, ichannels, ochannels,
                upsample=i > 0, upsample_rgb=i < (len(self.channels) - 1),
            ) for i, (ochannels, ichannels) in enumerate(iochannels[::-1])
        ])

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor,
        h: torch.Tensor = None,
        f: torch.Tensor = None,
    ) -> torch.Tensor:
        residual = x
        x = self.preprocess(x)

        affines: List[torch.Tensor] = []
        for i, (down, affine) in enumerate(zip(self.downsample, self.affine)):
            x = down(x)
            affines.append(affine(x))
            # if i == n: x = torch.cat([x, h], dim=1)

        # torch.cat([x, f], dim=1)
        rgb = None
        for i, up in enumerate(self.upsample):
            affine = affines[len(affines) - i - 1]
            x, rgb = up(x, rgb, style, noise)

        return residual + rgb


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.channels = [16, 32, 64, 128, 512]
        self.channels_f1 = [512 * 2, 1024, 1024, 1024, 1024, 1024]

        iochannels = list(zip(
            [self.channels[0]] + self.channels[:-1], self.channels,
        ))
        iochannels_f1 = list(zip(
            [self.channels_f1[0]] + self.channels_f1[:-1], self.channels_f1,
        ))

        self.preprocess = nn.Conv2d(
            3, self.channels[0], kernel_size=3, padding=(3 - 1) // 2,
        )
        
        self.downsample = nn.ModuleList([
            pt2_blocks.DownsampleBlock(ichannels, ochannels, downsample=i > 0)
            for i, (ichannels, ochannels) in enumerate(iochannels)
        ])

        self.downsample_f1 = nn.ModuleList([
            pt2_blocks.DownsampleBlock(ichannels, ochannels, downsample=i > 0)
            for i, (ichannels, ochannels) in enumerate(iochannels_f1)
        ])

        self.logits = nn.Conv2d(self.channels_f1[-1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        
        for downsample in self.downsample:
            x = downsample(x)
        x = torch.cat([x, f], dim=1)
        
        for downsample in self.downsample_f1:
            x = downsample(x)
        
        x = self.logits(x).view(x.size(0), -1)
        
        return x


if __name__ == '__main__':
    x = torch.rand((2, 3, 512, 512))
    z = torch.rand((2, 32))
    n = torch.rand((2, 1, 512, 512))
    f = torch.rand((2, 512, 32, 32))

    out = Generator(latent_dim=32)(x, z, n)
    pred = Discriminator()(out, f)