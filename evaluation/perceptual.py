from beautifultable import BeautifulTable
from paintstorch.network import Generator, Illustration2Vec
from torch.utils.data import DataLoader
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
LPIPS = lpips.LPIPS(net="alex", spatial=True).cuda()

datasets = {
    "NH": NoHintDataset(images, preps),
    **{f"SH {r}": SparseHintsDataset(images, preps, r) for r in [4, 16, 64]},
    "FH": FullHintsDataset(images, preps),
}
ds = {name: 0 for name in datasets.keys()}

with torch.no_grad():
    for name, dataset in datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        d_list = []
        for y, x, h, _ in tqdm(loader, desc=f"FID {name}"):
            h = F.interpolate(h, size=(128, 128))
            y, x, h = y.cuda(), x.cuda(), h.cuda()
            y_, *_ = G(x, h, F1(x))
            d_list.append(LPIPS(y, y_).cpu().numpy())
            
        ds[name] = np.mean(d_list)

mean = np.mean(list(ds.values()))
table = BeautifulTable()
table.columns.header = [f"LPIPS {name}" for name in ds.keys()] + ["Mean"]
table.rows.append([f"{d * 100:.2f}" for d in list(ds.values()) + [mean]])
print(table)
