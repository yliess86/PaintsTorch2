from beautifultable import BeautifulTable
from itertools import chain, product
from paintstorch.data import isimg, isnpz, listdir, Sample, xDoG
from paintstorch.network import Generator, Illustration2Vec
from PIL import Image
from skimage.draw import disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import argparse
import lpips
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class Dataset_(Dataset):
    def __init__(self, images: str, preps: str) -> None:
        self.xdog = xDoG()
        self.normalize = T.Normalize((0.5, ) * 3, (0.5, ) * 3)

        images = chain(*(listdir(f) for f in listdir(images)))
        self.images = sorted(list(filter(isimg, images)))

        preps = chain(*(listdir(f) for f in listdir(preps)))
        self.preps = sorted(list(filter(isnpz, preps)))

    def __len__(self) -> int:
        return len(self.images)


class NoHintDataset(Dataset_):
    def __init__(self, images: str, preps: str) -> None:
        super(NoHintDataset, self).__init__(images, preps)

    def __getitem__(self, idx: int) -> Sample:
        image = Image.open(self.images[idx]).convert("RGB").resize((512, 512))
        image = np.array(image) / 255
        
        mask = np.ones((*image.shape[:2], 1))
        lineart = np.repeat(self.xdog(image)[:, :, None], 3, axis=-1)

        y = image
        x = np.concatenate([lineart, mask], axis=-1)
        h = np.zeros((*image.shape[:2], 4))

        y = torch.from_numpy(y).permute((2, 0, 1)).float()
        x = torch.from_numpy(x).permute((2, 0, 1)).float()
        h = torch.from_numpy(h).permute((2, 0, 1)).float()

        y = self.normalize(y)
        x[:3] = self.normalize(x[:3])
        h[:3] = self.normalize(h[:3])

        return Sample(y, x, h, y)


class SparseHintsDataset(Dataset_):
    def __init__(self, images: str, preps: str, res: int = 8) -> None:
        super(SparseHintsDataset, self).__init__(images, preps)
        self.res = res
        self.positions = list(product(
            range(512 // self.res, 512, 512 // self.res),
            range(512 // self.res, 512, 512 // self.res),
        ))
        self.sizes = [
            5 if i % 2 == 0 else 10 for i in range(len(self.positions))
        ]

    def __getitem__(self, idx: int) -> Sample:
        image = Image.open(self.images[idx]).convert("RGB").resize((512, 512))
        image = np.array(image) / 255

        mask = np.ones((*image.shape[:2], 1))
        lineart = np.repeat(self.xdog(image)[:, :, None], 3, axis=-1)

        colors = np.load(self.preps[idx])["colors"]
        colors = Image.fromarray((colors * 255).astype(np.uint8))
        colors = colors.resize((512, 512))
        colors = np.array(colors) / 255

        hints = np.zeros((*image.shape[:2], 4))
        for position, radius in zip(self.positions, self.sizes):
            rr, cc = disk(position, radius=radius, shape=image.shape[:2])
            hints[rr, cc, -1] = 1.0
            hints[rr, cc, :3] = colors[position]
        
        y = image
        x = np.concatenate([lineart, mask], axis=-1)
        h = hints

        y = torch.from_numpy(y).permute((2, 0, 1)).float()
        x = torch.from_numpy(x).permute((2, 0, 1)).float()
        h = torch.from_numpy(h).permute((2, 0, 1)).float()

        y = self.normalize(y)
        x[:3] = self.normalize(x[:3])
        h[:3] = self.normalize(h[:3])

        return Sample(y, x, h, y)


class FullHintsDataset(Dataset_):
    def __init__(self, images: str, preps: str) -> None:
        super(FullHintsDataset, self).__init__(images, preps)

    def __getitem__(self, idx: int) -> Sample:
        image = Image.open(self.images[idx]).convert("RGB").resize((512, 512))
        image = np.array(image) / 255

        mask = np.ones((*image.shape[:2], 1))
        lineart = np.repeat(self.xdog(image)[:, :, None], 3, axis=-1)

        hints = np.load(self.preps[idx])["colors"]
        hints = Image.fromarray((hints * 255).astype(np.uint8))
        hints = hints.resize((512, 512))
        hints = np.array(hints) / 255
        
        y = image
        x = np.concatenate([lineart, mask], axis=-1)
        h = np.concatenate([hints, mask], axis=-1)

        y = torch.from_numpy(y).permute((2, 0, 1)).float()
        x = torch.from_numpy(x).permute((2, 0, 1)).float()
        h = torch.from_numpy(h).permute((2, 0, 1)).float()

        y = self.normalize(y)
        x[:3] = self.normalize(x[:3])
        h[:3] = self.normalize(h[:3])

        return Sample(y, x, h, y)


parser = argparse.ArgumentParser()
parser.add_argument("--features",      type=int, default=32)
parser.add_argument("--batch_size",    type=int, default=4)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--dataset",       type=str, default="./images_full")
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
