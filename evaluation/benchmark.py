from beautifultable import BeautifulTable
from paintstorch.network import Generator, Illustration2Vec
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from tqdm import tqdm
from evaluation.data import (
    chainer_postprod, FullHintsDataset, NoHintDataset, SparseHintsDataset,
)

import lpips
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionV3Features(nn.Module):
    def __init__(self) -> None:
        super(InceptionV3Features, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True)

        def hook(module: nn.Module, _, output: torch.Tensor) -> None:
            module.register_buffer("mixed_7c", output)
        self.inception_v3.Mixed_7c.register_forward_hook(hook)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inception_v3(F.interpolate(x, size=(299, 299)))
        features = self.inception_v3.Mixed_7c.mixed_7c
        features = F.adaptive_avg_pool2d(features, (1, 1))
        return features.view(x.size(0), 2048)

    @staticmethod
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


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--features",      type=int, default=32)
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--num_workers",   type=int, default=4)
    parser.add_argument("--dataset",       type=str, default="./dataset")
    parser.add_argument("--model",         type=str, required=True)
    parser.add_argument("--bn",            action="store_true")
    args = parser.parse_args()

    images = args.dataset
    preps = f"{(images[:-1] if images.endswith('/') else images)}_preprocessed"

    ckpt = torch.load(args.model)

    G = nn.DataParallel(Generator(args.features, bn=args.bn))
    G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)
    G = G.module.eval().cuda()

    x = torch.zeros((2, 4, 512, 512))
    F1 = torch.jit.trace(Illustration2Vec("./models/i2v.pth").eval(), x).cuda()
    
    I = InceptionV3Features().eval().cuda()
    L = lpips.LPIPS(net="alex", spatial=True).cuda()

    PC = torch.jit.load("./models/paintschainer.ts").cuda()

    amounts = [4, 16, 64]
    exp_datasets = {
        "NH": NoHintDataset(images, preps),
        **{f"SH {r}": SparseHintsDataset(images, preps, r) for r in amounts},
        "FH": FullHintsDataset(images, preps),
    }
    exp_names = exp_datasets.keys()

    benchmark = {
        "FID": {
            "PaintsTorch2": {name: 0 for name in exp_names},
            "PaintsChainer": {name: 0 for name in exp_names},
        },
        "LPIPS": {
            "PaintsTorch2": {name: 0 for name in exp_names},
            "PaintsChainer": {name: 0 for name in exp_names},
        }
    }
    metric_names = benchmark.keys()
    net_names = benchmark["FID"].keys()

    for net_name in net_names:
        chainer = net_name == "PaintsChainer"
        with torch.no_grad():
            for exp_name, exp_dataset in exp_datasets.items():
                exp_dataset.chainer = chainer
                loader = DataLoader(
                    exp_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )

                desc = f"{exp_name} {net_name}"
                fid_real, fid_fake = [], []
                d_list = []

                if chainer:
                    for y, x1, x2, _ in tqdm(loader, desc=desc):
                        y, x1, x2 = y.cuda(), x1.cuda(), x2.cuda()

                        y_ = PC(x1, x2)
                        y_ = chainer_postprod(y_)
                        y  = chainer_postprod(y)

                        fid_real.append(I(y).cpu().numpy())
                        fid_fake.append(I(y_).cpu().numpy())
                        d_list.append(L(y, y_).cpu().numpy())
                else:
                    for y, x, h, _ in tqdm(loader, desc=desc):
                        y, x, h = y.cuda(), x.cuda(), h.cuda()

                        h = F.interpolate(h, size=(128, 128))
                        y_ = G(x, h, F1(x))[0]
                    
                        fid_real.append(I(y).cpu().numpy())
                        fid_fake.append(I(y_).cpu().numpy())
                        d_list.append(L(y, y_).cpu().numpy())

                benchmark["LPIPS"][net_name][exp_name] = np.mean(d_list)
                benchmark["FID"][net_name][exp_name] = InceptionV3Features.fid(
                    np.concatenate(fid_real), np.concatenate(fid_fake),
                )

    for metric_name in metric_names:
        table = BeautifulTable()
        table.columns.header = (
            ["Model"] + [f"{metric_name} {n}" for n in exp_names] + ["Mean"]
        )

        for net_name in net_names:
            values = list(benchmark[metric_name][net_name].values())
            mean = np.mean(values)
            table.rows.append(
                [net_name] + [f"{f:.2f}" for f in values] + [mean]
            )
        print(table)