from PIL import Image
from torch.utils.data import Dataset
from typing import Any, NamedTuple

import numpy as np
import os
import torchvision.transforms as T


class File(NamedTuple):
    artist: str
    illustration: str


class Illustration(NamedTuple):
    artist_id: int
    illustration: Image.Image


class PaintsTorch2Dataset(Dataset):
    def __init__(self, path: str, is_train: bool = False) -> None:
        super(PaintsTorch2Dataset, self).__init__()
        self.path = path
        self.transforms = T.Compose([
            T.RandomRotation(360),
            T.Resize(512),
            T.RandomCrop((512, 512)),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        ]) if is_train else T.Compose([T.Resize((512, 512))])
        
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
        illustration = self.transforms(illustration)
        return Illustration(artist_id, illustration)

    def style(self, artist_id: int) -> Image.Image:
        files = [
            f for f in self.files if self.artist2id[f.artist] == artist_id
        ]

        file = files[np.random.choice(range(len(files)))]
        illustration = Image.open(file.illustration).convert("RGB")
        illustration = self.transforms(illustration)
        return illustration