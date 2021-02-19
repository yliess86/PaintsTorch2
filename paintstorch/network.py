from collections import OrderedDict
from torchvision.models import vgg16
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16Features(nn.Module):
    def __init__(self) -> None:
        super(VGG16Features, self).__init__()
        model = vgg16(pretrained=True)
        model.features = nn.Sequential(*list(model.features.children())[:9])
        self.model = model.features
        
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) - 0.5
        self.register_buffer("mean", mean)
        
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :3]
        return self.model((x.mul(0.5) - self.mean) / self.std)


class Illustration2Vec(nn.Module):
    def __init__(self, path: str) -> None:
        super(Illustration2Vec, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1539, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=True),
        )
        model.load_state_dict(torch.load(path))
        model = nn.Sequential(*list(model.children())[:15])
        self.model = model

        mean = torch.Tensor([164.76139251, 167.47864617, 181.13838569])
        mean = mean.view(1, 3, 1, 1)
        self.register_buffer("mean", mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :3]
        x = F.avg_pool2d(x, 2, 2)
        x = x.mul(0.5).add(0.5).mul(255)
        x = self.model(x - self.mean)
        return x


class ConvBn2d(nn.Sequential):
    def __init__(self, 
        ic: int,
        oc: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(ConvBn2d, self).__init__(OrderedDict(
            conv=nn.Conv2d(
                ic,
                oc,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            bn=nn.BatchNorm2d(oc),
        ))


class ResNetXtBottleneck(nn.Module):
    def __init__(
        self,
        ic: int,
        oc: int,
        cardinality: int,
        stride: int = 1,
        dilation: int = 1,
        bn: bool = True,
    ) -> None:
        super(ResNetXtBottleneck, self).__init__()
        conv = ConvBn2d if bn else nn.Conv2d
        self.reduce = conv(ic, oc // 2, kernel_size=1)
        self.conv = conv(
            oc // 2,
            oc // 2,
            kernel_size=2 + stride,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=cardinality,
        )
        self.expand = conv(oc // 2, oc, kernel_size=1)
        self.shortcut = (
            nn.AvgPool2d(2, stride=stride) if stride > 1 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.reduce(x), 0.2, True)
        h = F.leaky_relu(self.conv(h), 0.2, True)
        h = F.leaky_relu(self.expand(h), 0.2, True)
        return h + self.shortcut(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        ic: int,
        hc: int,
        oc: int,
        cardinality: int,
        dilations: List[int],
        bn: bool = True,
    ) -> None:
        super(DecoderBlock, self).__init__()
        conv = ConvBn2d if bn else nn.Conv2d
        self.preprocess = conv(ic, hc, kernel_size=3, padding=1)
        self.process = nn.ModuleList([
            ResNetXtBottleneck(
                hc, hc, cardinality=cardinality, dilation=d, bn=bn,
            ) for d in dilations
        ])
        self.postprocess = nn.Sequential(
            conv(hc, oc, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor = None,
    ) -> torch.Tensor:
        if residual is not None:
            x = torch.cat([residual, x], 1)
        x = F.leaky_relu(self.preprocess(x), 0.2, True)
        for process in self.process:
            x = F.leaky_relu(process(x), 0.2, True)
        x = F.leaky_relu(self.postprocess(x), 0.2, True)
        return x


class Generator(nn.Module):
    def __init__(self, features: int = 64, bn: bool = True) -> None:
        super(Generator, self).__init__()
        conv = ConvBn2d if bn else nn.Conv2d

        f0, f1, f2, f4, f8 = [int(features * f) for f in [0.5, 1, 2, 4, 8]]
        fh = f2 + f1               # Hints Injection
        ff = f8 + 512              # Features Injection
        c0, c1 = f1 // 4, f1 // 2  # Cardinality

        self.convh = conv(4, f1, 7, padding=3)
        
        self.conv1 = conv(4, f0, 3, padding=1)
        self.conv2 = conv(f0, f1, 4, stride=2, padding=1)
        self.conv3 = conv(f1, f2, 4, stride=2, padding=1)
        self.conv4 = conv(fh, f4, 4, stride=2, padding=1)
        self.conv5 = conv(f4, f8, 4, stride=2, padding=1)

        dilations5 = [1 for _ in range(20)]
        dilations4 = [1, 1, 2, 2, 4, 4, 2, 1]
        dilations3 = [1, 1, 2, 2, 4, 4, 2, 1]
        dilations2 = [1, 2, 4, 2, 1]

        self.deconv5 = DecoderBlock(ff, f8, f8 * 2, c1, dilations5, bn=bn)
        self.deconv4 = DecoderBlock(f4 * 2, f4, f4 * 2, c1, dilations4, bn=bn)
        self.deconv3 = DecoderBlock(f2 * 2, f2, f2 * 2, c1, dilations3, bn=bn)
        self.deconv2 = DecoderBlock(f1 * 2, f1, f1 * 2, c0, dilations2, bn=bn)
        self.out = conv(f1, 3, 3, padding=1)

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, f: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        h = F.leaky_relu(self.convh(h), 0.2, True)

        x1 = F.leaky_relu(self.conv1(x), 0.2, True)
        x2 = F.leaky_relu(self.conv2(x1), 0.2, True)
        x3 = F.leaky_relu(self.conv3(x2), 0.2, True)
        x4 = F.leaky_relu(self.conv4(torch.cat([x3, h], 1)), 0.2, True)
        x5 = F.leaky_relu(self.conv5(x4), 0.2, True)
        
        x = self.deconv5(f, x5)
        guided_x = x
        x = self.deconv4(x, x4)
        x = self.deconv3(x, x3)
        x = self.deconv2(x, x2)
        x = torch.tanh(self.out(torch.cat([x, x1], 1)))
        
        return x, guided_x, [x1, x2, x3, x4, x5]


class Guide(nn.Module):
    def __init__(self, features: int = 64, bn: bool = True) -> None:
        super(Guide, self).__init__()
        conv = ConvBn2d if bn else nn.Conv2d

        f1, f2, f4, f8 = [int(features * f) for f in [1, 2, 4, 8]]
        c1, c0 = f1 // 2, f1 // 4

        dilations4 = [1, 1, 2, 2, 4, 4, 2, 1]
        dilations3 = [1, 1, 2, 2, 4, 4, 2, 1]
        dilations2 = [1, 2, 1]

        self.deconv4 = DecoderBlock(f4 * 2, f4, f4 * 2, c1, dilations4, bn=bn)
        self.deconv3 = DecoderBlock(f2 * 2, f2, f2 * 2, c1, dilations3, bn=bn)
        self.deconv2 = DecoderBlock(f1 * 2, f1, f1 * 2, c0, dilations2, bn=bn)
        self.out = conv(f1, 3, kernel_size=3, padding=1)

    def forward(
        self, x: torch.Tensor, residuals: List[torch.Tensor],
    ) -> torch.Tensor:
        x1, x2, x3, x4, _ = residuals
        x = self.deconv4(x, x4)
        x = self.deconv3(x, x3)
        x = self.deconv2(x, x2)
        x = torch.tanh(self.out(torch.cat([x, x1], 1)))
        return x


class Discriminator(nn.Module):
    def __init__(self, features: int = 64, bn: bool = True) -> None:
        super(Discriminator, self).__init__()
        conv = ConvBn2d if bn else nn.Conv2d

        f1, f2, f4, f8 = [int(features * f) for f in [1, 2, 4, 8]]
        ff = f4 + 512  # Features Injection
        c1 = f1 // 8   # Cardinality

        self.conv1 = conv(3, f1, 7, padding=3)
        self.conv2 = conv(f1, f1, 4, stride=2, padding=1)

        self.bottleneck1 = nn.Sequential(
            ResNetXtBottleneck(f1, f1, c1, dilation=1, bn=bn),
            ResNetXtBottleneck(f1, f1, c1, stride=2, dilation=1, bn=bn),
        )
        self.conv3 = conv(f1, f2, 1)
        
        self.bottleneck2 = nn.Sequential(
            ResNetXtBottleneck(f2, f2, c1, dilation=1, bn=bn),
            ResNetXtBottleneck(f2, f2, c1, stride=2, dilation=1, bn=bn),
        )
        self.conv4 = conv(f2, f4, 1)
        
        self.bottleneck3 = nn.Sequential(
            ResNetXtBottleneck(f4, f4, c1, dilation=1, bn=bn),
            ResNetXtBottleneck(f4, f4, c1, stride=2, dilation=1, bn=bn),
        )
        self.conv5 = conv(ff, f8, 3, padding=1)
        
        self.bottleneck4 = nn.Sequential(*[
            ResNetXtBottleneck(f8, f8, c1, stride=s, dilation=1, bn=bn)
            for s in [1, 2, 1, 2, 1, 2, 1]
        ])
        self.conv6 = conv(f8, f8, 4)
        
        self.classifier = nn.Linear(f8, 1)

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.conv2(x), 0.2, True)
        
        x = self.bottleneck1(x)
        x = F.leaky_relu(self.conv3(x), 0.2, True)

        x = self.bottleneck2(x)
        x = F.leaky_relu(self.conv4(x), 0.2, True)
        
        x = self.bottleneck3(x)
        x = F.leaky_relu(self.conv5(torch.cat([x, f], 1)), 0.2, True)
        
        x = self.bottleneck4(x)
        x = F.leaky_relu(self.conv6(x), 0.2, True)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
