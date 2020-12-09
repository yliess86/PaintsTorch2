from paintstorch2.data.color.base import ColorSimplifier
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple

import numpy as np


Range = Tuple[int, int]


class kMeansColorSimplifier:
    def __init__(self, colors: Range) -> None:
        super(kMeansColorSimplifier, self).__init__()
        self.colors = colors

    def __call__(self, img: Image.Image, *args, **kwargs) -> Image.Image:
        x = np.array(img) / 255
        h, w, c = x.shape 

        x = x.reshape((h * w, c))
        kmeans = KMeans(n_clusters=int(np.random.uniform(*self.colors)))
        kmeans.fit(x)
        
        labels = kmeans.predict(x)
        x = kmeans.cluster_centers_[labels]
        x = x.reshape((h, w, c))

        return Image.fromarray((x * 255).astype(np.uint8))


if __name__ == '__main__':
    from io import BytesIO

    import os
    import requests


    BASE = "https://upload.wikimedia.org/wikipedia/en"
    URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
    
    img = Image.open(BytesIO(requests.get(URL).content)).resize((512,) * 2)
    color = kMeansColorSimplifier((5, 10))(img)

    img.show()
    color.show()