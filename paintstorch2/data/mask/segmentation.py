from paintstorch2.data.mask.base import MaskGenerator
from paintstorch2.data.segmentation import get_regions
from paintstorch2.model import SKELETONIZER
from PIL import Image
from typing import Tuple

import numpy as np
import torch


Range = Tuple[int, int]


class SegmentationMaskGenerator(MaskGenerator):
    def __init__(self, mix: Range) -> None:
        super(SegmentationMaskGenerator, self).__init__()
        self.skeletonizer = torch.jit.load(SKELETONIZER)
        self.mix = mix

    def __call__(self, img: Image.Image, *args, **kwargs) -> Image.Image:
        mix = np.random.randint(*self.mix)
        x = np.array(img) / 255.0

        with torch.no_grad():
            in_x = torch.Tensor(x).permute((2, 0, 1))
            skeleton = self.skeletonizer(in_x[None, ...])[0, 0].numpy()

        regions = get_regions(skeleton * 255)
        selection_indices = np.random.choice(range(len(regions)), size=(mix, ))
        
        mask = np.zeros(x.shape[:2], dtype=np.uint8)
        for idx in selection_indices:
            mask[regions[idx]] = 255

        img = Image.fromarray(mask)
        return img