from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from numba import njit
from numba.core.errors import (
    NumbaDeprecationWarning, NumbaPendingDeprecationWarning,
)
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import disk
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, NamedTuple, Tuple

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import warnings
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


isimg = lambda f: f.split(".")[-1].lower() in ["png", "jpg", "jpeg"]
isnpz = lambda f: f.split(".")[-1].lower() in ["npz"]
listdir = lambda p: [os.path.join(p, f) for f in os.listdir(p)]
same_artist = lambda a, b: a.split("/")[-2] == b.split("/")[-2]


class Rangei(NamedTuple):
    a: int
    b: int


class Rangef(NamedTuple):
    a: float
    b: float


def thinning(fill_map: np.ndarray, max_iter: int = 100) -> np.ndarray:
    line_id = 0
    h, w = fill_map.shape[:2]
    result = fill_map.copy()

    for iter in range(max_iter):
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
def count_all(labeled: np.ndarray, all_counts: List[int]) -> None:
    M, N = labeled.shape
    for x in range(M):
        for y in range(N):
            i = labeled[x, y] - 1
            if i > -1:
                all_counts[i] = all_counts[i] + 1
    return


@njit
def trace_all(
    labeled: np.ndarray,
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    cs: List[int],
) -> None:
    M, N = labeled.shape
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
    normal = normal.clip(0, 255).astype(np.uint8)
    
    marker[marker > 100] = 255
    marker[marker < 255] = 0
    
    labels, _ = label(marker / 255)
    labels = labels.astype(np.int32)
    
    water = cv2.watershed(normal, labels)
    water = thinning(water + 1)
    
    return find_all(water)


def save_preprocessed(
    path: str, colors: np.ndarray, masks: np.ndarray,
) -> None:
    np.savez(path, colors=colors, masks=masks)


def load_preprocessed(path: str) -> Tuple[np.ndarray, np.ndarray]:
    loader = np.load(path)
    return loader["colors"], loader["masks"]


def preprocess(model: str, f: str, dest: str) -> int:
    *_, artist, img = f.split("/")
    folder = os.path.join(dest, artist)
    os.makedirs(folder, exist_ok=True)

    npz = f"{img.split('.')[-2]}.npz"
    path = os.path.join(folder, npz)
    if os.path.isfile(path):
        return 1

    segmentor = Segmentor(model)

    img = Image.open(f).convert("RGB")
    size = img.size

    img = np.array(img.resize((512, 512))) / 255
    colors, masks = segmentor(img)
    colors = cv2.resize((colors * 255).astype(np.uint8), dsize=size)
    masks = cv2.resize((masks * 255).astype(np.uint8), dsize=size)

    segmentor.cpu()
    del segmentor

    save_preprocessed(path, colors, masks)
    return 1


class Segmentor:
    def __init__(self, path: str) -> None:
        self.model = torch.jit.load(path).eval()

    def cuda(self) -> Segmentor:
        self.model = self.model.cuda()
        return self

    def cpu(self) -> Segmentor:
        self.model = self.model.cpu()
        return self

    @torch.no_grad()
    def __call__(
        self, img: np.ndarray, max_colors: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        device = next(self.model.parameters()).device
        inp = torch.Tensor(img).permute((2, 0, 1)).unsqueeze(0).to(device)

        skeleton = self.model(inp)
        skeleton = skeleton.squeeze(0)[0].cpu().numpy()
        regions = list(filter(len, get_regions(skeleton * 255.0)))

        colors = np.zeros((*img.shape[:2], 3))
        for r, region in enumerate(regions):
            y = np.clip(region[0], 0, img.shape[0] - 1)
            x = np.clip(region[1], 0, img.shape[1] - 1)
            if len(img[y, x]):
                colors[y, x] = [np.median(img[y, x][..., i]) for i in range(3)]

        H, W, C = colors.shape
        colors = colors.reshape((H * W, C))
        kmeans = KMeans(n_clusters=max_colors)
        labels = kmeans.fit_predict(colors)
        colors = kmeans.cluster_centers_[labels]
        colors = colors.reshape((H, W, C))

        labels = labels.reshape((H, W))
        masks = np.zeros((*img.shape[:2], kmeans.n_clusters))
        for label in range(kmeans.n_clusters):
            masks[labels == label, label] = 1

        return colors, masks

    @classmethod
    def preprocess(cls, model: str, root: str, dest: str) -> None:
        files = chain(*(filter(isimg, listdir(f)) for f in listdir(root)))
        files = list(files)

        pbar = tqdm(total=len(files), desc="Preprocess")
        with ProcessPoolExecutor() as executor:
            for f in files:
                process = executor.submit(preprocess, model, f, dest)
                pbar.update(process.result())

    @classmethod
    def sample(
        cls, img: np.ndarray, masks: np.ndarray, n_color: float,
    ) -> np.ndarray:
        n = int(masks.shape[-1] * n_color)
        if n_color <= 0 or n_color >=1:
            return np.ones(img.shape[:2], dtype=np.float32)

        idxs = np.random.choice(masks.shape[-1], size=(n, ))
        return np.max(masks[:, :, idxs], axis=-1)


class xDoG:
    def __init__(
        self,
        γ: float = 0.95,
        ϕ: float = 1e9,
        ϵ: float = -1e1,
        k: float = 4.5,
        σ: float = 0.3,
    ) -> None:
        self.γ = γ
        self.ϕ = ϕ
        self.ϵ = ϵ
        self.k = k
        self.σ = σ

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = (img[..., 0] + img[..., 1] + img[..., 2]) / 3
        
        gaussian_a = gaussian_filter(x, self.σ)
        gaussian_b = gaussian_filter(x, self.σ * self.k)

        dog = gaussian_a - self.γ * gaussian_b

        inf = dog < self.ε
        xdog = inf * 1 + ~inf * (1 - np.tanh(self.φ * dog))

        xdog -= xdog.min()
        xdog /= xdog.max()
        xdog = xdog >= threshold_otsu(xdog)
        xdog = 1 - xdog
        
        return xdog


def compose(
    img: np.ndarray, lineart: np.ndarray, mask: np.ndarray, black: float = 1.0,
) -> np.ndarray:
    lineart[lineart == 0] = 1 - black
    res = img * (1 - mask)[..., None] + (lineart * mask)[..., None]
    return res


def generate_input(composition: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ret = np.zeros((*composition.shape[:2], 4))
    ret[..., :3] = composition
    ret[..., -1] = mask
    return ret


def generate_hints(
    colors: np.ndarray,
    mask: np.ndarray,
    n_color: float,
    radius_range: Rangei = Rangei(2, 32 + 1),
    proportion_range: Rangef = Rangef(1e-4, 1e-2),
) -> np.ndarray:
    if n_color <= 0:
        return np.zeros((*colors.shape[:2], 4))
        
    elif n_color >= 1:
        return np.concatenate([
            colors, np.ones((*colors.shape[:2], 1)),
        ], axis=-1)
    
    coverage = np.sum(mask == 1) / np.prod(colors.shape[:2])
    proportion = np.random.uniform(*proportion_range) * coverage

    activations = np.random.rand(*colors.shape[:2])
    samples = np.random.random_sample(size=colors.shape[:2])
    interest = (proportion * activations >= samples) * mask
    
    hints = np.zeros((*colors.shape[:2], 4), dtype=np.float32)
    for position in zip(*np.nonzero(interest)):
        radius = np.random.randint(*radius_range)
        rr, cc = disk(position, radius=radius, shape=colors.shape[:2])
        hints[rr, cc, -1] = 1.0
        hints[rr, cc, :3] = colors[position]
    
    return hints


class Sample(NamedTuple):
    y: torch.Tensor  # (3, H,   W  ) Target Illustration
    x: torch.Tensor  # (4, H,   W  ) Composition Input
    h: torch.Tensor  # (4, H/4, W/4) Color Hints
    c: torch.Tensor  # (3, H,   W  ) Segmented Color Map
    

class PaintsTorchDataset(Dataset):
    def __init__(
        self,
        paths: Tuple[str, str],
        segmentor: str,
        black_range: Rangei = Rangei(6, 10 + 1),
        n_color_range: Rangei = Rangei(0, 10 + 1),
        train: bool = True,
        curriculum: bool = False,
    ) -> None:
        super(PaintsTorchDataset, self).__init__()
        self.black_range = black_range
        self.n_color_range = n_color_range
        self.train = train

        self.curriculum = curriculum if self.train else False
        self.curriculum_state = 1.0
        
        self.normalize = T.Normalize((0.5, ) * 3, (0.5, ) * 3)
        self.xdog = xDoG()

        imgs, prep = paths
        if not os.path.isdir(prep):
            os.makedirs(prep, exist_ok=True)
        Segmentor.preprocess(segmentor, imgs, prep)
        
        self.imgs = chain(*(filter(isimg, listdir(f)) for f in listdir(imgs)))
        self.imgs = sorted(list(self.imgs))

        self.prep = chain(*(filter(isnpz, listdir(f)) for f in listdir(prep)))
        self.prep = sorted(list(self.prep))
    
    def __len__(self) -> int:
        return len(self.imgs)

    def transform(self, *imgs: List) -> List:
        if not self.train:
            transforms = T.Compose([T.Resize(512), T.CenterCrop(512)])
            return [transforms(img) for img in imgs]

        imgs = [T.Resize(512)(img) for img in imgs]

        i, j, h, w = T.RandomCrop.get_params(imgs[0], (512, 512))
        imgs = [TF.crop(img, i, j, h, w) for img in imgs]

        hflip = np.random.rand() > 0.5
        imgs = [TF.hflip(img) if hflip else img for img in imgs]

        vflip = np.random.rand() > 0.5
        imgs = [TF.vflip(img) if vflip else img for img in imgs]

        angle = T.RandomRotation.get_params((0, 360))
        imgs = [TF.rotate(img, angle) for img in imgs]

        jitter = T.ColorJitter.get_params(
            (.6, 1.4), (.6, 1.4), (.6, 1.4), (-.1, .1),
        )
        imgs = [jitter(img) if img.mode == "RGB" else img for img in imgs]

        return imgs

    @property
    def n_color(self) -> int:
        if not self.train:
            return 0

        n_color = (
            np.random.randint(*self.n_color_range) / (self.n_color_range - 1)
        )
        if self.curriculum:
            n_color = int(max(self.curriculum_state, n_color))

        return n_color

    @property
    def black(self) -> int:
        if not self.train:
            return 1

        black = np.random.randint(*self.black_range) / (self.black_range - 1)
        return black

    def __getitem__(self, idx: int) -> Sample:
        paint = self.imgs[idx]
        colors = self.prep[idx]
        
        paint = Image.open(paint).convert("RGB") 
        colors, masks = load_preprocessed(colors)
        colors = Image.fromarray(colors).convert("RGB")
        masks = [
            Image.fromarray(masks[..., i]).convert("L")
            for i in range(masks.shape[-1])
        ]
        paint, colors, *masks = self.transform(paint, colors, *masks)

        paint = np.array(paint) / 255
        colors = np.array(colors) / 255
        masks = np.transpose([
            np.array(mask) for mask in masks
        ], axes=(1, 2, 0)) / 255

        lineart = self.xdog(paint)
        mask = Segmentor.sample(paint, masks, n_color=self.n_color)
        composition = compose(paint, lineart, mask, black=self.black)
        input = generate_input(composition, mask)
        hints = generate_hints(colors, mask, self.n_color)

        y = torch.from_numpy(paint).permute((2, 0, 1))
        x = torch.from_numpy(input).permute((2, 0, 1))
        h = torch.from_numpy(hints).permute((2, 0, 1))
        c = torch.from_numpy(colors).permute((2, 0, 1))

        y = self.normalize(y)
        x[:3] = self.normalize(x[:3])
        h[:3] = self.normalize(h[:3])
        c = self.normalize(c)

        return Sample(y.float(), x.float(), h.float(), c.float())
