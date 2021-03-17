from itertools import chain, product
from paintstorch.data import isimg, isnpz, listdir, Sample, xDoG
from PIL import Image
from skimage.draw import disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T


class Dataset_(Dataset):
    def __init__(self, images: str, preps: str, chainer: bool = False) -> None:
        self.xdog = xDoG()
        self.normalize = T.Normalize((0.5, ) * 3, (0.5, ) * 3)
        self.chainer = chainer

        images = chain(*(listdir(f) for f in listdir(images)))
        self.images = sorted(list(filter(isimg, images)))

        preps = chain(*(listdir(f) for f in listdir(preps)))
        self.preps = sorted(list(filter(isnpz, preps)))

    def __len__(self) -> int:
        return len(self.images)

    def transform(
        self, y: torch.Tensor, x: torch.Tensor, h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.chainer:
            y, x, h = y * 255, x * 255, h * 255
            x = x.mean(0).unsqueeze(0)
            h = h.permute((1, 2, 0))

            x1 = np.zeros((h.size(0), h.size(1), 4))
            x1[:, :, 0] = x[0].numpy()
            x1[:, :, 1] = -512
            x1[:, :, 2] = 128
            x1[:, :, 3] = 128

            h = h.numpy().astype(np.uint8)
            r, g, b, a = cv2.split(h)
            h = cv2.cvtColor(cv2.merge((b, g, r)), cv2.COLOR_BGR2YUV)

            s = a != 0
            x1[s, 1:] = h[s]
            x1 = torch.from_numpy(x1.transpose(2, 0, 1)).float()

            return y, x1, x

        else:
            y = self.normalize(y)
            x[:3] = self.normalize(x[:3])
            h[:3] = self.normalize(h[:3])
            
            return y, x, h


class NoHintDataset(Dataset_):
    def __init__(self, images: str, preps: str, chainer: bool = False) -> None:
        super(NoHintDataset, self).__init__(images, preps, chainer)

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

        y, x, h = self.transform(y, x, h)

        return Sample(y, x, h, y)


class SparseHintsDataset(Dataset_):
    def __init__(
        self, images: str, preps: str, res: int = 8, chainer: bool = False,
    ) -> None:
        super(SparseHintsDataset, self).__init__(images, preps, chainer)
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

        y, x, h = self.transform(y, x, h)

        return Sample(y, x, h, y)


class FullHintsDataset(Dataset_):
    def __init__(self, images: str, preps: str, chainer: bool = False) -> None:
        super(FullHintsDataset, self).__init__(images, preps, chainer)

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

        y, x, h = self.transform(y, x, h)

        return Sample(y, x, h, y)


def apply(*objects: List, transform: Callable) -> List:
    return [transform(o) for o in objects]


def squeeze_permute_cpu_np(*objects: List) -> List:
    return apply(
        *objects,
        transform=lambda x: x.squeeze(0).permute((1, 2, 0)).cpu().numpy(),
    )