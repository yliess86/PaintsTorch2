from paintstorch2.data.color.base import ColorSimplifier
from paintstorch2.data.dataset.base import PaintsTorch2Dataset
from paintstorch2.data.hints.base import HintsGenerator
from paintstorch2.data.lineart.base import LineartGenerator
from paintstorch2.data.mask.base import MaskGenerator
from typing import NamedTuple

import torch
import torchvision.transforms as T
import yaml


class Modules(NamedTuple):
    color: ColorSimplifier
    hints: HintsGenerator
    lineart: LineartGenerator
    mask: MaskGenerator


class Data(NamedTuple):
    artist_id: int
    composition: torch.Tensor
    hints: torch.Tensor
    style: torch.Tensor
    illustration: torch.Tensor
    

class ModularPaintsTorch2Dataset(PaintsTorch2Dataset):
    def __init__(
        self, modules: Modules, path: str, is_train: bool = False,
    ) -> None:
        super(ModularPaintsTorch2Dataset, self).__init__(path, is_train)
        self.modules = modules
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx: int) -> Data:
        data = super(ModularPaintsTorch2Dataset, self).__getitem__(idx)
        artist_id, illustration_512 = data

        style_512 = super(ModularPaintsTorch2Dataset, self).style(artist_id)
        colors_512 = self.modules.color(illustration_512)
        colors_128 = colors_512.resize((128, 128))
        hints_128 = self.modules.hints(colors_128)
        lineart_512 = self.modules.lineart(illustration_512)
        mask_512 = self.modules.mask(illustration_512)

        illustration = self.to_tensor(illustration_512)
        style = self.to_tensor(style_512)
        lineart = self.to_tensor(lineart_512)
        mask = self.to_tensor(mask_512)
        
        hints = torch.cat([
            self.to_tensor(hints_128.hints),
            self.to_tensor(hints_128.mask),
        ], dim=0)

        composition = torch.cat([
            illustration * (1 - mask) + lineart * mask,
            mask,
        ], dim=0)

        return Data(artist_id, composition, hints, style, illustration)

    @classmethod
    def from_config(
        cls, config: str, path: str, is_train: bool = False,
    ) -> "ModularPaintsTorch2Dataset":
        from paintstorch2.data import (
            COLOR_SIMPLIFIERS,
            HINTS_GENERATORS,
            LINEART_GENERATORS,
            MASK_GENERATORS,
        )
        
        with open(config, "r") as f:
            content = f.read()
        data = yaml.load(content)

        color_data = data["color"]
        color = COLOR_SIMPLIFIERS[color_data["name"]](
            **color_data.get("params", {})
        )

        hints_data = data["hints"]
        hints = HINTS_GENERATORS[hints_data["name"]](
            **hints_data.get("params", {})
        )

        lineart_data = data["lineart"]
        lineart = LINEART_GENERATORS[lineart_data["name"]](
            **lineart_data.get("params", {})
        )
        
        mask_data = data["mask"]
        mask = MASK_GENERATORS[mask_data["name"]](
            **mask_data.get("params", {})
        )

        modules = Modules(color, hints, lineart, mask)
        return cls(modules, path, is_train)