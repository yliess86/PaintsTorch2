from paintstorch2.data.hints.base import Hints, HintsGenerator
from PIL import Image, ImageDraw

import numpy as np


class RandomHintsGenerator(HintsGenerator):
    def __init__(self, p: float = 5e-3, size: int = 2) -> None:
        super(RandomHintsGenerator, self).__init__()
        self.p = p
        self.size = size

    def __call__(self, img: Image.Image) -> Hints:
        activation = np.random.random((img.height, img.width)) <= self.p
        
        colors = Image.new("RGB", size=(img.width, img.height))
        draw_colors = ImageDraw.Draw(colors)
        
        mask = Image.new("L", size=(img.width, img.height))
        draw_mask = ImageDraw.Draw(mask)
        
        for y in range(activation.shape[0]):
            for x in range(activation.shape[1]):
                if activation[y, x]:
                    draw_colors.rectangle((
                        (x - self.size // 2, y - self.size // 2),
                        (x + self.size // 2, y + self.size // 2),
                    ), fill=tuple(np.array(img)[y, x]), outline=None, width=0)

                    draw_mask.rectangle((
                        (x - self.size // 2, y - self.size // 2),
                        (x + self.size // 2, y + self.size // 2),
                    ), fill="white", outline=None, width=0)

        return Hints(colors, mask)


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