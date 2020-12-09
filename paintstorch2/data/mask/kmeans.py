from paintstorch2.data.mask.base import MaskGenerator
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from typing import Tuple

import numpy as np


Range = Tuple[int, int]


class kMeansMaskGenerator(MaskGenerator):
    def __init__(self, colors: Range) -> None:
        super(kMeansMaskGenerator, self).__init__()
        self.colors = colors

    def __call__(self, img: Image.Image) -> Image.Image:
        x = np.array(img) / 255
        h, w, c = x.shape 

        x = x.reshape((h * w, c))
        k = int(np.random.uniform(*self.colors))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        labels = kmeans.predict(x)

        mask = np.zeros((h * w))
        mask[labels == np.random.randint(k)] = 1
        mask = mask.reshape((h, w))
        mask = gaussian_filter(mask, 2) > 0.5

        return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((512,) * 2)
    mask = kMeansMaskGenerator((5, 15))(img)

    img.show()
    mask.show()