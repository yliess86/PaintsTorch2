from typing import List

import numpy as np
import paintstorch2.model.blocks as pt2_blocks
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Generator(nn.Module):
#     def __init__(self, latent_dim: int) -> None:
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.channels = [32, 64, 128, 512]
        
#         iochannels = list(zip([
#             self.channels[0]] + self.channels[:-1], self.channels,
#         ))

#         self.preprocess = nn.Conv2d(
#             3, self.channels[0], kernel_size=3, padding=(3 - 1) // 2,
#         )

#         self.downsample = nn.ModuleList([
#             pt2_blocks.DownsampleBlock(
#                 ichannels, ochannels, downsample=i > 0,
#             ) for i, (ichannels, ochannels) in enumerate(iochannels)
#         ])

#         self.affine = nn.ModuleList([
#             nn.Conv2d(channel, channel, kernel_size=1, bias=False)
#             for channel in self.channels
#         ])

#         self.upsample = nn.ModuleList([
#             pt2_blocks.UpsampleBlock(
#                 latent_dim, ichannels, ochannels,
#                 upsample=i > 0, upsample_rgb=i < (len(self.channels) - 1),
#             ) for i, (ochannels, ichannels) in enumerate(iochannels[::-1])
#         ])

#     def forward(
#         self,
#         x: torch.Tensor,
#         style: torch.Tensor,
#         noise: torch.Tensor,
#         h: torch.Tensor = None,
#         f: torch.Tensor = None,
#     ) -> torch.Tensor:
#         residual = x
#         x = self.preprocess(x)

#         affines: List[torch.Tensor] = []
#         for i, (down, affine) in enumerate(zip(self.downsample, self.affine)):
#             x = down(x)
#             affines.append(affine(x))
#             # if i == n: x = torch.cat([x, h], dim=1)

#         # torch.cat([x, f], dim=1)
#         rgb = None
#         for i, up in enumerate(self.upsample):
#             affine = affines[len(affines) - i - 1]
#             x, rgb = up(x, rgb, style, noise)

#         return residual + rgb


class Discriminator(nn.Module):
    def __init__(self, capacity: int = 64) -> None:
        super(Discriminator, self).__init__()
        self.capacity = capacity
        c1, c2, c4, c8 = [capacity * i for i in [1, 2, 4, 8]]
        cf = c4 + 512

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            pt2_blocks.ResNetXtBootleneck(c1, c1, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c1, c1, cardinality=8, stride=2),
            nn.Conv2d(c1, c2, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            pt2_blocks.ResNetXtBootleneck(c2, c2, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c2, c2, cardinality=8, stride=2),
            nn.Conv2d(c2, c4, kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            pt2_blocks.ResNetXtBootleneck(c4, c4, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c4, c4, cardinality=8, stride=2),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(cf, c8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            pt2_blocks.ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),

            nn.Conv2d(c8, c8, kernel_size=4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.logits = nn.Conv2d(c8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.encoder2(torch.cat([x, f], dim=1))
        x = self.logits(x)
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    x = torch.rand((2, 3, 512, 512))
    z = torch.rand((2, 32))
    n = torch.rand((2, 1, 512, 512))
    f = torch.rand((2, 512, 32, 32))

    # out = Generator(latent_dim=32)(x, z, n)
    pred = Discriminator()(x, f)
    print(pred.size())