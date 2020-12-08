from paintstorch2.data.lineart.base import LineartGenerator
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu

import numpy as np


class xDoGLineartGenerator(LineartGenerator):
    def __init__(
        self,
        γ: float = 0.95,
        ϕ: float = 1e9,
        ϵ: float = -1e1,
        k: float = 4.5,
        σ: float = 0.3,
    ) -> None:
        super(LinearGenerator, self).__init__()
        self.γ = γ
        self.ϕ = ϕ
        self.ϵ = ϵ
        self.k = k
        self.σ = σ

    def __call__(self, img: Image.Image) -> Image.Image:
        x = np.array(img.convert("L")) / 255
        
        gaussian_a = gaussian_filter(x, self.σ)
        gaussian_b = gaussian_filter(x, self.σ * self.k)

        dog = gaussian_a - self.γ * gaussian_b

        inf = dog < self.ε
        xdog = inf * 1 + ~inf * (1 - np.tanh(self.φ * dog))

        xdog -= xdog.min()
        xdog /= xdog.max()
        xdog = xdog >= threshold_otsu(xdog)
        xdog = 1 - xdog
        
        return Image.fromarray((xdog * 255).astype(np.uint8))


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((512,) * 2)
    lineart = xDoGLineartGenerator()(img)

    img.show()
    lineart.show()