from typing import List

import numpy as np
import paintstorch2.model.blocks as pt2_blocks
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Embedding(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super(Embedding, self).__init__()
        self.latent_dim = latent_dim
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(4096, latent_dim)

        self.register_buffer("mean",
            torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) - 0.5,
        )
        self.register_buffer("std",
            torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg16(((x * 0.5) - self.mean) / self.std)


class Generator(nn.Module):
    def __init__(
        self, latent_dim: int, capacity: int = 64, ichannels: int = 4,
    ) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        c0 = capacity // 2
        c1, c2, c4, c8 = [capacity * i for i in [1, 2, 4, 8]]
        ch = c2 + c1   # Hints Encoding Injection
        cf = c8 + 512  # Illustration2Vec Injection

        self.hints_peprocess = nn.Sequential(
            nn.Conv2d(4, c1, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder0 = nn.Sequential(
            nn.Conv2d(ichannels, c0, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder1 = nn.Sequential(
            nn.Conv2d(c0, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(ch, c4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(c4, c8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder4 = pt2_blocks.UpsampleBlock(latent_dim, cf, c8)
        self.decoder3 = pt2_blocks.UpsampleBlock(latent_dim, c8 + c4, c4)
        self.decoder2 = pt2_blocks.UpsampleBlock(latent_dim, c4 + c2, c2)
        self.decoder1 = pt2_blocks.UpsampleBlock(latent_dim, c2 + c1, c1)
        
        self.out = nn.Conv2d(c1 + c0, 3, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        f: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hints_peprocess(h)
        
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(torch.cat([x2, h], dim=1))
        x4 = self.encoder4(x3)

        x, rgb = self.decoder4(torch.cat([x4, f], dim=1), None, style, noise)
        x, rgb = self.decoder3(torch.cat([x, x3], dim=1), rgb, style, noise)
        x, rgb = self.decoder2(torch.cat([x, x2], dim=1), rgb, style, noise)
        x, _ = self.decoder1(torch.cat([x, x1], dim=1), rgb, style, noise)

        x = self.out(torch.cat([x, x0], dim=1))

        return x


class Discriminator(nn.Module):
    def __init__(self, capacity: int = 64) -> None:
        super(Discriminator, self).__init__()
        self.capacity = capacity
        c1, c2, c4, c8 = [capacity * i for i in [1, 2, 4, 8]]
        cf = c4 + 512  # Illustration2Vec Injection

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