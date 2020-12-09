from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import numpy as np
import os
import paintstorch2.data.color as pt2_color
import paintstorch2.data.hints as pt2_hints
import paintstorch2.data.lineart as pt2_lineart
import paintstorch2.data.mask as pt2_mask
import requests


def compose_input(
    img: Image.Image,
    lineart: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    img = np.array(img) / 255
    lineart = np.array(lineart) / 255
    mask = np.array(mask) / 255
    
    img *= (1 - mask)[:, :, None]
    lineart *= mask
    composition = img + lineart[:, :, None]

    return Image.fromarray((composition * 255).astype(np.uint8))


BASE = "https://upload.wikimedia.org/wikipedia/en"
URL = os.path.join(BASE, "7/7d/Lenna_%28test_image%29.png")
SIZE_512 = (512, ) * 2
SIZE_64 = ( 64, ) * 2

img_512 = Image.open(BytesIO(requests.get(URL).content)).resize(SIZE_512)
img_64 = img_512.resize(SIZE_64)

lineart_512 = pt2_lineart.xDoGLineartGenerator()(img_512)

mask_512 = pt2_mask.PatchMaskGenerator((64, 256), (64, 256))(img_512)
mask_64 = mask_512.resize(SIZE_64)

colors_64 = pt2_color.kMeansColorSimplifier((5, 15))(img_64)
hints_64 = pt2_hints.RandomHintsGenerator()(colors_64)

fig = plt.figure(figsize=(3 * 4, 2 * 4), facecolor="white")

ax = fig.add_subplot(2, 3, 1)
ax.imshow(np.array(img_512))
ax.set_axis_off()

ax = fig.add_subplot(2, 3, 2)
ax.imshow(np.array(lineart_512), cmap="gray")
ax.set_axis_off()

ax = fig.add_subplot(2, 3, 3)
ax.imshow(np.array(compose_input(img_512, lineart_512, mask_512)))
ax.set_axis_off()

ax = fig.add_subplot(2, 3, 4)
ax.imshow(np.array(colors_64))
ax.set_axis_off()

ax = fig.add_subplot(2, 3, 5)
ax.imshow(np.array(mask_512), cmap="gray")
ax.set_axis_off()

ax = fig.add_subplot(2, 3, 6)
ax.imshow(np.array(hints_64.hints))
ax.set_axis_off()

fig.canvas.draw()
fig.subplots_adjust(hspace=0.05, wspace=0.05)

plt.show()