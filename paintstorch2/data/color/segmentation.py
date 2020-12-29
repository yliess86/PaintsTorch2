from paintstorch2.data.color.base import ColorSimplifier
from paintstorch2.data.segmentation import get_regions
from paintstorch2.model import SKELETONIZER
from PIL import Image

import numpy as np
import torch


class SegmentationColorSimplifier(ColorSimplifier):
    def __init__(self) -> None:
        super(SegmentationColorSimplifier, self).__init__()
        self.skeletonizer = torch.jit.load(SKELETONIZER)

    def __call__(self, img: Image.Image, *args, **kwargs) -> Image.Image:
        x = np.array(img) / 255.0

        with torch.no_grad():
            in_x = torch.Tensor(x).permute((2, 0, 1))
            skeleton = self.skeletonizer(in_x[None, ...])[0, 0].numpy()

        regions = get_regions(skeleton * 255)
        colors = np.zeros((*x.shape[:2], 3))
        for region in regions:
            colors[region] = np.array([
                np.median(x[region][..., i]) for i in range(3)
            ])

        img = Image.fromarray((colors * 255).astype(np.uint8))
        return img


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((128,) * 2)
    color = SegmentationColorSimplifier()(img)

    img.show()
    color.show()