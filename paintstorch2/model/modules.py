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
        upsample: int = None,
    ) -> None:
        super(ToRGB, self).__init__()
        self.style_mapping = nn.Linear(latent_dim, ichannels)
        self.mod_conv2d = ModConv2D(
            ichannels, outchannels, kernel_size=1, demod=False,
        )
        
        self.upsample = nn.Upsample(
            scale_factor=upsample,
            mode="bilinear",
            align_corners=False
        ) if upsample is not None else None

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, style: torch.Tensor,
    ) -> torch.Tensor:
        x = self.mod_conv2d(x, self.style_mapping(style))
        
        if residual is not None:
            x = x + residual
        if self.upsample is not None:
            x = self.upsample(x)
        
        return x


if __name__ == "__main__":
    x = torch.rand((2, 3, 64, 64))
    y = torch.rand((2, 3))
    z = torch.rand((2, 32))

    out = ModConv2D(3, 8, kernel_size=3)(x, y)
    out = ToRGB(32, 8, upsample=2)(out, x, z)