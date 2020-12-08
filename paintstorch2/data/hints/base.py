from PIL import Image
from typing import NamedTuple


class Hints(NamedTuple):
    hints: Image.Image
    mask: Image.Image


class HintsGenerator:
    def __call__(self, img: Image.Image, *args, **kwargs) -> Hints:
        raise NotImplementedError("__call__ method must be implemented.")