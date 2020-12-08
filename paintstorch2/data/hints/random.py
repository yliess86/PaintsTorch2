from paintstorch2.data.hints.base import Hints, HintsGenerator
from PIL import Image, ImageDraw

import numpy as np


class RandomHintsGenerator(HintsGenerator):
    def __init__(self, p: float = 5e-3) -> None:
        super(RandomHintsGenerator, self).__init__()
        self.p = p

    def __call__(self, img: Image.Image) -> Hints:
        size = (img.height, img.width)
        mask = np.random.random(size) <= self.p

        x = np.array(img) / 255 * mask[:, :, None]

        return Hints(
            Image.fromarray((x * 255).astype(np.uint8)),
            Image.fromarray((mask * 255).astype(np.uint8)),
        )


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((64,) * 2)
    hints, mask = RandomHintsGenerator()(img)

    img.show()
    hints.show()
    mask.show()