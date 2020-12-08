from PIL import Image


class MaskGenerator:
    def __call__(self, img: Image.Image, *args, **kwargs) -> Image.Image:
        raise NotImplementedError("__call__ method must be implemented.")