from torchvision.models import inception_v3

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV3Features(nn.Module):
    FEATURES = 2048
    
    def __init__(self) -> None:
        super(InceptionV3Features, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True)
        self.inception_v3.Mixed_7c.register_forward_hook(self.hook)

    def hook(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor,
    ) -> None:
        self.mixed_7c = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inception_v3(F.interpolate(x, size=(299, 299)))
        features = F.adaptive_avg_pool2d(self.mixed_7c, (1, 1))
        return features.view(x.size(0), self.FEATURES)


def fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    μ_real = np.mean(real_features, axis=0)
    μ_fake = np.mean(fake_features, axis=0)

    Σ_real = np.cov(real_features, rowvar=False)
    Σ_fake = np.cov(fake_features, rowvar=False)

    Δ = (μ_real - μ_fake)
    Δ_squared = Δ.dot(Δ)

    π = Σ_real.dot(Σ_fake)
    π_sqrt, _ = scipy.linalg.sqrtm(π, disp=False)
    if np.iscomplexobj(π_sqrt):
        π_sqrt = π_sqrt.real
    π_trace = np.trace(π_sqrt)
    
    return Δ_squared + np.trace(Σ_real) + np.trace(Σ_fake) - 2 * π_trace