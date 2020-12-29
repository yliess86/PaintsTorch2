from paintstorch2.data.dataset.base import Data
from torch.utils.data import Dataset
from typing import List

import numpy as np
import os
import torch
import torchvision.transforms as T


class PreprocessedPaintsTorch2Dataset(Dataset):
    def __init__(self, path: str, is_train: bool = False) -> None:
        super(PreprocessedPaintsTorch2Dataset, self).__init__()
        self.files = [os.path.join(path, file) for file in os.listdir(path)]
        self.files = list(sorted(self.files))
        self.is_train = is_train

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

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        artist_id, *data = torch.load(self.files[idx])
        composition, hints, style, illustration = self.flip(self.rotate(data))
        composition[:3], hints[:3], illustration = self.jitter([
            composition[:3], hints[:3], illustration,
        ])
        style,  = self.jitter([style, ])
        return Data(artist_id, composition, hints, style, illustration)