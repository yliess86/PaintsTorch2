from PIL import Image
from torch.utils.data import Dataset
from typing import Any, NamedTuple

import numpy as np
import os
import torch
import torchvision.transforms as T


class File(NamedTuple):
    artist: str
    illustration: str


class Illustration(NamedTuple):
    artist_id: int
    illustration: Image.Image


class Data(NamedTuple):
    artist_id: int
    composition: torch.Tensor
    hints: torch.Tensor
    illustration: torch.Tensor


class IllustrationsDataset(Dataset):
    def __init__(self, path: str) -> None:
        super(IllustrationsDataset, self).__init__()
        self.path = path
        
        self.files = [
            File(artist, os.path.join(path, artist, f))
            for artist in sorted(os.listdir(path))
            for f in sorted(os.listdir(os.path.join(path, artist)))
        ]
        
        self.artist2id = {
            artist: i for i, artist in
            enumerate(sorted(set(artist for artist, _ in self.files)))
        }

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Illustration:
        file = self.files[idx]
        artist_id = self.artist2id[file.artist]
        illustration = Image.open(file.illustration).convert("RGB")
        return Illustration(artist_id, illustration)