from paintstorch2.data.color.base import ColorSimplifier
from PIL import Image
from typing import Tuple

import numpy as np


Range = Tuple[int, int]


class QuantizeColorSimplifier:
    def __init__(self, colors: Range) -> None:
        super(QuantizeColorSimplifier, self).__init__()
        self.colors = colors

    def __call__(self, img: Image.Image, *args, **kwargs) -> Image.Image:
        colors = int(np.random.uniform(*self.colors))
        return img.convert("P", palette=Image.ADAPTIVE, colors=colors)


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((512,) * 2)
    color = QuantizeColorSimplifier((5, 10))(img)

    img.show()
    color.show()