from paintstorch2.data.mask.base import MaskGenerator
from PIL import Image, ImageDraw
from typing import NamedTuple, Tuple

import numpy as np


Range = Tuple[int, int]


class Size(NamedTuple):
    w: int
    h: int


class Position(NamedTuple):
    x: int
    y: int


class PatchMaskGenerator(MaskGenerator):
    def __init__(self, width: Range, height: Range) -> None:
        super(PatchMaskGenerator, self).__init__()
        self.width = width
        self.height = height

    def sample_patch_size(self) -> Size:
        width = np.random.uniform(*self.width)
        height = np.random.uniform(*self.height)
        return Size(width, height)

    def sample_patch_position(
        self, patch_size: Size, src_size: Size,
    ) -> Position:
        x = np.random.uniform(src_size.w - patch_size.w)
        y = np.random.uniform(src_size.h - patch_size.h)
        return Position(x, y)

    def __call__(self, img: Image.Image) -> Image.Image:
        src_size = Size(img.width, img.height)

        patch_size = self.sample_patch_size()
        patch_pos = self.sample_patch_position(patch_size, src_size)
        
        top_left = Position(
            patch_pos.x - patch_size.w // 2,
            patch_pos.y - patch_size.h // 2,
        )
        bottom_right = Position(
            patch_pos.x + patch_size.w // 2,
            patch_pos.y + patch_size.h // 2,
        )

        mask = Image.new("L", src_size)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((top_left, bottom_right), fill="#ffffff")

        return mask


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((512,) * 2)
    mask = PatchMaskGenerator((64, 128), (64, 128))(img)

    img.show()
    mask.show()