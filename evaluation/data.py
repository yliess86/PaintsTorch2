from paintstorch.data import PaintsTorchDataset, Sample
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, List

import argparse
import numpy as np
import torch
import torch.nn as nn


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
parser.add_argument("--dataset", type=str, default="./images_full")
args = parser.parse_args()

images = args.dataset
preps = f"{(images[:-1] if images.endswith('/') else images)}_preprocessed"
skeletonizer = "./models/skeletonizer.ts"
dataset = PaintsTorchDataset((images, preps), skeletonizer, train=True)

Image.fromarray(np.vstack([
    stitch(dataset[0]) for _ in tqdm(range(20), desc="Mosaic")
])).show()