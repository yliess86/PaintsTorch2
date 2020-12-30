from torch.utils.data import Dataset
from tqdm import tqdm
from typing import NamedTuple, List

import numpy as np
import os
import torch
import torchvision.transforms as T


class Data(NamedTuple):
    artist_id: int
    composition: torch.Tensor
    hints: torch.Tensor
    style: torch.Tensor
    illustration: torch.Tensor


class PaintsTorch2Dataset(Dataset):
    def __init__(self, path: str, is_train: bool = False) -> None:
        super(PaintsTorch2Dataset, self).__init__()
        self.files = [os.path.join(path, file) for file in os.listdir(path)]
        self.files = list(sorted(self.files))
        self.is_train = is_train

        self.artist2files = {}
        for file in tqdm(self.files, desc="Referencing Styles"):
            artist_id, *_ = torch.load(file)
            if not artist_id in self.artist2files:
                self.artist2files[artist_id] = []
            self.artist2files[artist_id].append(file)

    def style(self, artist_id: int, file: str) -> str:
        if not self.is_train:
            return file

        return np.random.choice(self.artist2files[artist_id])

    def rotate(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.is_train:
            return imgs

        θ = np.random.randint(0, 360)
        return [T.functional.rotate(img, angle=θ) for img in imgs]

    def flip(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.is_train:
            return imgs
        
        if np.random.rand() > 0.5:
            imgs = [T.functional.vflip(img) for img in imgs]
        if np.random.rand() > 0.5:
            imgs = [T.functional.hflip(img) for img in imgs]
        return imgs

    def jitter(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.is_train:
            return imgs

        b = np.random.uniform(0, 0.2)
        imgs = [T.functional.adjust_brightness(img, b) for img in imgs]

        c = np.random.uniform(0, 0.2)
        imgs = [T.functional.adjust_contrast(img, c) for img in imgs]

        s = np.random.uniform(0, 0.2)
        imgs = [T.functional.adjust_saturation(img, s) for img in imgs]

        h = np.random.uniform(0, 0.1)
        imgs = [T.functional.adjust_hue(img, h) for img in imgs]

        return imgs

    def crop(self, imgs: List[torch.Tensor], size: int) -> List[torch.Tensor]:
        if not self.is_train:
            return [T.functional.center_crop(img, [size,] * 2) for img in imgs]

        return [
            T.functional.crop(
                img,
                np.random.randint(0, max(min(img.size(1) - size, size), 1)),
                np.random.randint(0, max(min(img.size(2) - size, size), 1)),
                size,
                size,
            )
            for img in imgs
        ]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        file = self.files[idx]
        artist_id, compo, hints, illu = torch.load(file)
        *_, style = torch.load(self.style(artist_id, file))

        compo, hints, illu = self.crop([compo, hints, illu], size=512)
        compo, hints, illu = self.flip(self.rotate([compo, hints, illu]))
        compo[:3], hints[:3], illu = self.jitter([compo[:3], hints[:3], illu])
        hints = T.functional.resize(hints, [128, 128])
        
        style, = self.crop([style,], size=512)
        style, = self.flip(self.rotate([style,]))
        style, = self.jitter([style,])
        
        return Data(artist_id, compo, hints, style, illu)