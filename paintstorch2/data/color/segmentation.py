from numba import njit
from paintstorch2.data.color.base import ColorSimplifier
from paintstorch2.model import SKELETONIZER
from PIL import Image
from scipy.ndimage import label
from typing import List, Tuple

import cv2
import numpy as np
import torch


def thinning(fillmap: np.ndarray, max_iter: int = 100) -> np.ndarray:
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()

    for iterNum in range(max_iter):
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        line_mask = np.full((h, w), 255, np.uint8)
        line_mask[line_points] = 0
        line_border_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
            anchor=(-1, -1),
            iterations=1
        ) - line_mask
        line_border_points = np.where(line_border_mask == 255)

        result_tmp = result.copy()
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            if x - 1 > 0 and result[y, x - 1] != line_id:
                result_tmp[y, x] = result[y, x - 1]
                continue
            if x - 1 > 0 and y - 1 > 0 and result[y - 1, x - 1] != line_id:
                result_tmp[y, x] = result[y - 1, x - 1]
                continue
            if y - 1 > 0 and result[y - 1, x] != line_id:
                result_tmp[y, x] = result[y - 1, x]
                continue
            if y - 1 > 0 and x + 1 < w and result[y - 1, x + 1] != line_id:
                result_tmp[y, x] = result[y - 1, x + 1]
                continue
            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y, x] = result[y, x + 1]
                continue
            if x + 1 < w and y + 1 < h and result[y + 1, x + 1] != line_id:
                result_tmp[y, x] = result[y + 1, x + 1]
                continue
            if y + 1 < h and result[y + 1, x] != line_id:
                result_tmp[y, x] = result[y + 1, x]
                continue
            if y + 1 < h and x - 1 > 0 and result[y + 1, x - 1] != line_id:
                result_tmp[y, x] = result[y + 1, x - 1]
                continue

        result = result_tmp.copy()

    return result


def topo_compute_normal(dist: np.ndarray) -> np.ndarray:
    c = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, +1]]))
    r = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1], [+1]]))
    h = np.zeros_like(c + r, dtype=np.float32) + 0.75
    normal_map = np.stack([h, r, c], axis=2)
    normal_map /= np.sum(normal_map ** 2.0, axis=2, keepdims=True) ** 0.5
    return normal_map


@njit
def count_all(labeled: np.ndarray, all_counts: np.ndarray) -> None:
    M, N = labeled.shape[:2]
    for x in range(M):
        for y in range(N):
            i = labeled[x, y] - 1
            if i > -1:
                all_counts[i] = all_counts[i] + 1
    return


@njit
def trace_all(
    labeled: np.ndarray, xs: np.ndarray, ys: np.ndarray, cs: np.ndarray,
) -> None:
    M, N = labeled.shape[:2]
    for x in range(M):
        for y in range(N):
            current_label = labeled[x, y] - 1
            if current_label > -1:
                current_label_count = cs[current_label]
                xs[current_label][current_label_count] = x
                ys[current_label][current_label_count] = y
                cs[current_label] = current_label_count + 1
    return


def find_all(labeled: np.ndarray) -> np.ndarray:
    hist_size = int(np.max(labeled))
    if hist_size == 0:
        return []

    all_counts = [0 for _ in range(hist_size)]
    count_all(labeled, all_counts)
    
    xs = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    ys = [np.zeros(shape=(item, ), dtype=np.uint32) for item in all_counts]
    cs = [0 for item in all_counts]
    trace_all(labeled, xs, ys, cs)
    
    filled_area = []
    for _ in range(hist_size):
        filled_area.append((xs[_], ys[_]))
    return filled_area


def get_regions(skeleton: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    marker = skeleton.copy()
    normal = topo_compute_normal(marker) * 127.5 + 127.5
    marker[marker > 100] = 255
    marker[marker < 255] = 0
    
    labels, _ = label(marker / 255)
    water = thinning(cv2.watershed(
        normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32),
    ) + 1)
    
    return find_all(water)


class SegmentationColorSimplifier:
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