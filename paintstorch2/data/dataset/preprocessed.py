from paintstorch2.data.dataset.base import Data
from torch.utils.data import Dataset

import numpy as np
import os
import torch


class PreprocessedPaintsTorch2Dataset(Dataset):
    def __init__(self, path: str) -> None:
        super(PreprocessedPaintsTorch2Dataset, self).__init__()
        self.files = {}
        for file in sorted(os.listdir(path)):
            i, j = tuple(map(int, file.split(".")[0].split("_")))
            if i not in self.files:
                self.files[i] = []
            self.files[i].append(os.path.join(path, file))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        files = self.files[idx]
        file = files[np.random.choice(len(files))]
        data = torch.load(file)
        return Data(*data)