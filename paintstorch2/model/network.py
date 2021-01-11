from torchvision.models import vgg16
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SN = lambda conv: nn.utils.spectral_norm(conv)


def AdaIN(
    c: torch.Tensor, γ: torch.Tensor, β: torch.Tensor, ε: float = 1e-6,
) -> torch.Tensor:
    B, C, H, W = c.size()
    c = c.view(B, C, -1)
    μ_c = torch.mean(c, axis=-1, keepdims=True)
    σ_c = torch.std(c, axis=-1, keepdims=True) + ε
    adain = β.view(B, C, 1) * (c - μ_c) / σ_c + γ.view(B, C, 1)
    return adain.view(B, C, H, W)


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
        
        self.reduce = SN(nn.Conv2d(
            ichannels, hchannels, kernel_size=1, stride=1, bias=False,
        ))

        self.conv = SN(nn.Conv2d(
            hchannels, hchannels,
            kernel_size=2 + stride,
            stride=stride,
            padding=dilate,
            dilation=dilate,
            groups=cardinality,
            bias=False,
        ))
        
        self.expand = SN(nn.Conv2d(
            hchannels, ochannels, kernel_size=1, stride=1, bias=False,
        ))
        
        self.shortcut = nn.AvgPool2d(2, stride=stride) if stride > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck = self.reduce(x)
        bottleneck = self.conv(bottleneck)
        bottleneck = self.expand(bottleneck)

        if self.shortcut is not None:
            bottleneck = bottleneck + self.shortcut(x)
        
        return bottleneck


class Embedding(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super(Embedding, self).__init__()
        self.latent_dim = latent_dim
        self.vgg16 = vgg16(pretrained=True)
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
        c1, c2, c4, c8, c16 = [capacity * i for i in [1, 2, 4, 8, 16]]
        ch = c2 + c1   # Hints Encoding Injection
        cf = c8 + 512  # Illustration2Vec Injection

        self.hints_peprocess = nn.Sequential(
            SN(nn.Conv2d(4, c1, kernel_size=7, stride=1, padding=3)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder0 = nn.Sequential(
            SN(nn.Conv2d(ichannels, c0, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder1 = nn.Sequential(
            SN(nn.Conv2d(c0, c1, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder2 = nn.Sequential(
            SN(nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder3 = nn.Sequential(
            SN(nn.Conv2d(ch, c4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder4 = nn.Sequential(
            SN(nn.Conv2d(c4, c8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder4_pre = nn.Sequential(
            SN(nn.Conv2d(cf, c8, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder4_adain = nn.Linear(latent_dim, c8 * 2)
        self.decoder4_tunnel = nn.Sequential(*[
            ResNetXtBootleneck(c8, c8, cardinality=32, dilate=1)
            for _ in range(20)
        ])
        self.decoder4_post = nn.Sequential(
            SN(nn.Conv2d(c8, c16, kernel_size=3, stride=1, padding=1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder3_pre = nn.Sequential(
            SN(nn.Conv2d(c8, c4, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder3_adain = nn.Linear(latent_dim, c4 * 2)
        self.decoder3_tunnel = nn.Sequential(*[
            ResNetXtBootleneck(c4, c4, cardinality=32, dilate=d)
            for d in [1, 1, 2, 2, 4, 4, 2, 1]
        ])
        self.decoder3_post = nn.Sequential(
            SN(nn.Conv2d(c4, c8, kernel_size=3, stride=1, padding=1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder2_pre = nn.Sequential(
            SN(nn.Conv2d(c4, c2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder2_adain = nn.Linear(latent_dim, c2 * 2)
        self.decoder2_tunnel = nn.Sequential(*[
            ResNetXtBootleneck(c2, c2, cardinality=32, dilate=d)
            for d in [1, 1, 2, 2, 4, 4, 2, 1]
        ])
        self.decoder2_post = nn.Sequential(
            SN(nn.Conv2d(c2, c4, kernel_size=3, stride=1, padding=1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder1_pre = nn.Sequential(
            SN(nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder1_adain = nn.Linear(latent_dim, c1 * 2)
        self.decoder1_tunnel = nn.Sequential(*[
            ResNetXtBootleneck(c1, c1, cardinality=16, dilate=d)
            for d in [1, 2, 4, 2, 1]
        ])
        self.decoder1_post = nn.Sequential(
            SN(nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.out = SN(nn.Conv2d(
            c1, 3, kernel_size=3, stride=1, padding=1,
        ))

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        f: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hints_peprocess(h)
        
        x0 = self.encoder0(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(torch.cat([x2, h], dim=1))
        x4 = self.encoder4(x3)

        x = self.decoder4_pre(torch.cat([x4, f], dim=1))
        adain = self.decoder4_adain(s)
        sep = adain.size(1) // 2
        x = AdaIN(x, adain[:, sep:], adain[:, :sep])
        x = self.decoder4_tunnel(x)
        x = self.decoder4_post(x)

        x = self.decoder3_pre(torch.cat([x3, x], dim=1))
        adain = self.decoder3_adain(s)
        sep = adain.size(1) // 2
        x = AdaIN(x, adain[:, sep:], adain[:, :sep])
        x = self.decoder3_tunnel(x)
        x = self.decoder3_post(x)

        x = self.decoder2_pre(torch.cat([x2, x], dim=1))
        adain = self.decoder2_adain(s)
        sep = adain.size(1) // 2
        x = AdaIN(x, adain[:, sep:], adain[:, :sep])
        x = self.decoder2_tunnel(x)
        x = self.decoder2_post(x)

        x = self.decoder1_pre(torch.cat([x1, x], dim=1))
        adain = self.decoder1_adain(s)
        sep = adain.size(1) // 2
        x = AdaIN(x, adain[:, sep:], adain[:, :sep])
        x = self.decoder1_tunnel(x)
        x = self.decoder1_post(x)

        x = self.out(torch.cat([x, x0], dim=1))

        return x


class Discriminator(nn.Module):
    def __init__(self, capacity: int = 64) -> None:
        super(Discriminator, self).__init__()
        self.capacity = capacity
        c1, c2, c4, c8 = [capacity * i for i in [1, 2, 4, 8]]
        cf = c4 + 512  # Illustration2Vec Injection

        self.encoder1 = nn.Sequential(
            SN(nn.Conv2d(
                3, c1, kernel_size=7, stride=1, padding=3, bias=False,
            )),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(
                c1, c1, kernel_size=4, stride=2, padding=1, bias=False,
            )),
            nn.LeakyReLU(0.2, inplace=True),

            ResNetXtBootleneck(c1, c1, cardinality=8, stride=1),
            ResNetXtBootleneck(c1, c1, cardinality=8, stride=2),
            SN(nn.Conv2d(c1, c2, kernel_size=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            ResNetXtBootleneck(c2, c2, cardinality=8, stride=1),
            ResNetXtBootleneck(c2, c2, cardinality=8, stride=2),
            SN(nn.Conv2d(c2, c4, kernel_size=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            ResNetXtBootleneck(c4, c4, cardinality=8, stride=1),
            ResNetXtBootleneck(c4, c4, cardinality=8, stride=2),
        )

        self.encoder2 = nn.Sequential(
            SN(nn.Conv2d(
                cf, c8, kernel_size=3, stride=1, padding=1, bias=False
            )),
            nn.LeakyReLU(0.2, inplace=True),

            ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=2),
            ResNetXtBootleneck(c8, c8, cardinality=8, stride=1),

            SN(nn.Conv2d(c8, c8, kernel_size=4, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.logits = SN(nn.Conv2d(c8, 1, kernel_size=1))

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.encoder2(torch.cat([x, f], dim=1))
        x = self.logits(x)
        return x.view(x.size(0), -1)