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
        demodulation: bool = True,
    ) -> None:
        super(ModConv2D, self).__init__()
        self.ichannels = ichannels
        self.ochannels = ochannels
        self.kernel_size = kernel_size
        self.demodulation = demodulation
        
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

        if self.demodulation:
            dim = 2, 3, 4
            d = torch.rsqrt((weights ** 2).sum(dim=dim, keepdims=True) + ε)
            weights = weights * d

        x = x.view(1, -1, h, w)
        _, _, *size = weights.size()
        weights = weights.view(-1, *size)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.view(-1, self.ochannels, h, w)


if __name__ == "__main__":
    x = torch.rand((2, 3, 64, 64))
    y = torch.rand((2, 3))

    out = ModConv2D(3, 8, kernel_size=3)(x, y)