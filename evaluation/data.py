from itertools import chain, product
from paintstorch.data import (
    isimg, isnpz, listdir, PaintsTorchDataset, Sample, xDoG,
)
from PIL import Image
from skimage.draw import disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Callable, List

import argparse
import numpy as np
import torch
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


def apply(*objects: List, transform: Callable) -> List:
    return [transform(o) for o in objects]


def squeeze_permute_cpu_np(*objects: List) -> List:
    return apply(
        *objects,
        transform=lambda x: x.squeeze(0).permute((1, 2, 0)).cpu().numpy(),
    )


@torch.no_grad()
def stitch(sample: Sample) -> np.ndarray:
    y, x, h, c = sample
    y, x, h, c = squeeze_permute_cpu_np(y, x, h, c)
    x, h, c = x[..., :3], h[..., :3], c[..., :3]
    img = (np.hstack([y, x, h, c]) * 0.5) + 0.5
    img = (img * 255).astype(np.uint8)
    return img


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="./dataset")
args = parser.parse_args()

images = args.dataset
preps = f"{(images[:-1] if images.endswith('/') else images)}_preprocessed"
skeletonizer = "./models/skeletonizer.ts"
dataset = PaintsTorchDataset((images, preps), skeletonizer, train=True)

Image.fromarray(np.vstack([
    stitch(dataset[0]) for _ in tqdm(range(20), desc="Mosaic")
])).show()