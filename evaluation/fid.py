from beautifultable import BeautifulTable
from paintstorch.network import Generator, Illustration2Vec
from torchvision.models import inception_v3
from tqdm import tqdm
from evaluation.data import FullHintsDataset, NoHintDataset, SparseHintsDataset

import argparse
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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

datasets = {
    "NH": NoHintDataset(images, preps),
    **{f"SH {r}": SparseHintsDataset(images, preps, r) for r in [4, 16, 64]},
    "FH": FullHintsDataset(images, preps),
}
fids = {name: 0 for name in datasets.keys()}

with torch.no_grad():
    for name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        fid_real, fid_fake = [], []
        for y, x, h, _ in tqdm(loader, desc=f"FID {name}"):
            h = F.interpolate(h, size=(128, 128))
            y, x, h = y.cuda(), x.cuda(), h.cuda()
            y_, *_ = G(x, h, F1(x))
            fid_real.append(I(y).cpu().numpy())
            fid_fake.append(I(y_).cpu().numpy())

        fids[name] = InceptionV3Features.fid(
            np.concatenate(fid_real),
            np.concatenate(fid_fake),
        )

mean = np.mean(list(fids.values()))
table = BeautifulTable()
table.columns.header = [f"FID {name}" for name in fids.keys()] + ["Mean"]
table.rows.append([f"{fid:.2f}" for fid in list(fids.values()) + [mean]])
print(table)
