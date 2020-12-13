import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientPenalty(nn.Module):
    def __init__(self, discriminator: nn.Module, λ: float) -> None:
        super(GradientPenalty, self).__init__()
        self.D = discriminator
        self.λ = λ

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        α = torch.rand((real.size(0), 1, 1, 1))
        if real.is_cuda:
            α = α.cuda()
        
        t = α * real - (1 - α) * fake
        t.requires_grad = True
        Dt = self.D(t, f)
        
        grads = torch.autograd.grad(
            outputs=Dt,
            inputs=t,
            grad_outputs=torch.ones_like(Dt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return self.λ * ((grads.norm(2, dim=1) - 1) ** 2).mean()