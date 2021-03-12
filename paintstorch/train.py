from itertools import chain
from paintstorch.data import PaintsTorchDataset, Sample
from paintstorch.network import (
    Discriminator, Generator, Guide, Illustration2Vec, VGG16Features,
)
from torch.cuda.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable, List

import argparse
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply(*objects: List, transform: Callable) -> List:
    return [transform(o) for o in objects]


def parallel(*objects: List) -> List:
    return apply(*objects, transform=nn.DataParallel)


def cuda(*objects: List) -> List:
    return apply(*objects, transform=lambda x: x.cuda())


def clone_unsqueeze_cuda(*objects: List) -> List:
    return apply(*objects, transform=lambda x: x.clone().unsqueeze(0).cuda())


def squeeze_permute_cpu_np(*objects: List) -> List:
    return apply(
        *objects,
        transform=lambda x: x.squeeze(0).permute((1, 2, 0)).cpu().numpy(),
    )


class GradientPenalty(nn.Module):
    def __init__(self, λ: float) -> None:
        super(GradientPenalty, self).__init__()
        self.λ = λ

    def forward(
        self,
        D: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
        f: torch.Tensor,
    ) -> torch.Tensor:
        α = torch.rand((real.size(0), 1, 1, 1), device=real.device)
        t = α * real.data - (1 - α) * fake.data
        t.requires_grad = True
        Dt = D(t, f)

        grads = torch.autograd.grad(
            outputs=Dt,
            inputs=t,
            grad_outputs=torch.ones_like(Dt),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grads = grads.view(grads.size(0), -1)
        return self.λ * ((grads.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
@autocast()
def test(
    F1: nn.Module, G: nn.Module, C: nn.Module, sample: Sample, is_guide: bool,
) -> np.ndarray:
    y_, x_, h_, c_ = sample
    y, x, h, c = clone_unsqueeze_cuda(y_, x_, h_, c_)
    h_ = F.interpolate(h.clone(), size=(128, 128))
    f = F1(x)

    mask = x[:, -1].unsqueeze(1)
    fake, guide, residuals = G(x, h_, f)
    fake = x[:, :3] * (1 - mask) + fake.clip(-1, 1) * mask
    
    if is_guide:
        guide = c * (1 - mask) + C(guide, residuals).clip(-1, 1) * mask
    else:
        guide = c

    x, h, c, y, fake, guide = squeeze_permute_cpu_np(x, h, c, y, fake, guide)
    x, h, c = x[..., :3], h[..., :3], c[..., :3]
    
    img = (np.hstack([x, h, c, guide, y, fake]) * 0.5) + 0.5
    img = (img * 255).astype(np.uint8)
    return img


parser = argparse.ArgumentParser()
parser.add_argument("--features",      type=int, default=32)
parser.add_argument("--epochs",        type=int, default=40)
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--dataset",       type=str, default="./dataset")
parser.add_argument("--tensorboard",   type=str, default="./experiments")
parser.add_argument("--checkpoint",    type=str, default="./")
parser.add_argument("--guide",         action="store_true")
parser.add_argument("--parallel",      action="store_true")
parser.add_argument("--bn",            action="store_true")
parser.add_argument("--curriculum",    action="store_true")
args, _ = parser.parse_known_args()

epochs = args.epochs
batch_size = args.batch_size
features = args.features

α = 1e-4
β = 0.5, 0.9
ε_drift = 1e-3
λ1 = 1e-4
λ2 = 10
γ = 0.1
γ_step = epochs // 2
ε_curriculum_step = 0.9 / int(0.8 * epochs)

images = args.dataset
preps = f"{(images[:-1] if images.endswith('/') else images)}_preprocessed"

skeletonizer = "./models/skeletonizer.ts"
timestamp = datetime.datetime.now().timestamp()
log_dir = os.path.join(args.tensorboard, f"paintstorch_2_{timestamp}")

testset = PaintsTorchDataset((images, preps), skeletonizer, train=False)
sample_no_hints = testset[0]

dataset = PaintsTorchDataset((images, preps), skeletonizer, train=True)
sample_hints = dataset[0]

currciculum_state_ref = [0.9]
dataset.curriculum = args.curriculum
dataset.curriculum_state = currciculum_state_ref

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)

G = Generator(features, bn=args.bn)
D = Discriminator(features, bn=args.bn)
C = Guide(features, bn=args.bn) if args.guide else nn.Identity()

fake = torch.zeros((1, 4, 512, 512))
F1 = torch.jit.trace(Illustration2Vec("./models/i2v.pth").eval(), fake)
F2 = torch.jit.trace(VGG16Features().eval(), fake)

MSE = nn.MSELoss()
L1 = nn.L1Loss()
GP = GradientPenalty(λ2)

if args.parallel:
    G, D, C, F1, F2, MSE, GP = parallel(G, D, C, F1, F2, MSE, GP)
G, D, C, F1, F2, MSE, GP = cuda(G, D, C, F1, F2, MSE, GP)

optim = lambda x: AdamW(x, lr=α, betas=β)
optimG = optim(chain(G.parameters(), C.parameters()))
optimD = optim(D.parameters())

params = {"step_size": γ_step, "gamma": γ}
schedulerG = StepLR(optimG, **params)
schedulerD = StepLR(optimD, **params)

writer = SummaryWriter(log_dir=log_dir)
if not os.path.isdir(args.checkpoint):
    os.makedirs(args.checkpoint, exist_ok=True)

for epoch in tqdm(range(args.epochs), desc="Epoch"):
    total_D_fake = 0.0
    total_D_real = 0.0
    total_G_adv = 0.0
    total_G_content = 0.0
    total_G_guide = 0.0
    total_loss = 0.0

    G.train()
    D.train()
    with tqdm(loader, desc="Batch") as pbar:
        for y, x, h, c in pbar:
            h = F.interpolate(h, size=(128, 128))
            y, x, h, c = cuda(y, x, h, c)

            optimG.zero_grad()
            optimD.zero_grad()

            with torch.no_grad():
                f = F1(x)
            mask = x[:, -1].unsqueeze(1)
            fake, guide, residuals = G(x, h, f)
            fake = x[:, :3] * (1 - mask) + fake * mask
            if args.guide:
                guide = c * (1 - mask) + C(guide, residuals) * mask

            𝓛_adv = λ1 * torch.relu(1 - D(fake, f)).mean()
            𝓛_content = MSE(F2(fake), F2(y)).mean()
            𝓛_guide = L1(guide, c).mean() if args.guide else 0
            𝓛_G = 𝓛_adv + 𝓛_content + 𝓛_guide
            𝓛_G.backward()
            optimG.step()

            total_G_adv += 𝓛_adv.item() / len(loader)
            total_G_content += 𝓛_content.item() / len(loader)
            total_G_guide += 𝓛_guide.item() / len(loader) if args.guide else 0

            optimD.zero_grad()

            𝓛_fake = torch.relu(1 + D(fake.detach(), f)).mean()
            𝓛_real = torch.relu(1 - D(y, f)).mean()
            𝓛_real = 𝓛_real + ε_drift * 𝓛_real ** 2
            𝓛_p = GP(D, y, fake, f).mean()
            𝓛_D = 𝓛_fake + 𝓛_real + 𝓛_p
            𝓛_D.backward()
            optimD.step()
            
            total_D_fake += 𝓛_fake.item() / len(loader)
            total_D_real += 𝓛_real.item() / len(loader)

            total_D = total_D_fake + total_D_real
            total_G = total_G_adv + total_G_content + total_G_guide
            total_loss = total_D + total_G

            pbar.set_postfix(
                D_fake=total_D_fake,
                D_real=total_D_real,
                G_adv=total_G_adv,
                G_content=total_G_content,
                G_guide=total_G_guide,
                total=total_loss,
            )

    schedulerG.step()
    schedulerD.step()

    if args.curriculum:
        cur_state = currciculum_state_ref.pop()
        new_state = max(0, cur_state - ε_curriculum_step)
        currciculum_state_ref.append(new_state)

    G.eval()
    nh_img = test(F1, G, C, sample_no_hints, is_guide=args.guide)
    h_img = test(F1, G, C, sample_hints, is_guide=args.guide)

    step = epoch + 1
    
    writer.add_scalar("Loss/D/fake", total_D_fake, step)
    writer.add_scalar("Loss/D/real", total_D_real, step)
    writer.add_scalar("Loss/G/adv", total_G_adv, step)
    writer.add_scalar("Loss/G/content", total_G_content, step)
    writer.add_scalar("Loss/G/guide", total_G_guide, step)
    writer.add_scalar("Loss/total", total_loss, step)

    writer.add_scalar("Hypermarameter/lr", optimG.param_groups[0]['lr'], step)
    if args.curriculum:
        writer.add_scalar("Hypermarameter/curriculum", cur_state, step)
    
    writer.add_image("Image/no_hints", nh_img, step, dataformats="HWC")
    writer.add_image("Image/hints", h_img, step, dataformats="HWC")

    torch.save(
        G.state_dict(),
        os.path.join(args.checkpoint, f"checkpoint_{epoch:02d}.pth"),
    )
