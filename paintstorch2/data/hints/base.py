from PIL import Image
from typing import List


class HintGenerator:
    def __call__(self, img: Image, *args, **kwargs) -> List[Image]:
        raise NotImplementedError("__call__ method must be implemented.")