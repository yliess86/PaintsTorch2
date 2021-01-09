from paintstorch2.data.color.base import ColorSimplifier
from paintstorch2.data.dataset.illustration import Data, IllustrationsDataset
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
    

class ModularDataset(IllustrationsDataset):
    def __init__(self, modules: Modules, path: str) -> None:
        super(ModularDataset, self).__init__(path)
        self.modules = modules
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx: int) -> Data:
        artist_id, _illu = super(ModularDataset, self).__getitem__(idx)

        _illu_512 = T.Resize(512)(_illu)
        _illu_512_512 = T.Resize((512, 512))(_illu_512)
        h, w = _illu_512.height, _illu_512.width

        _colors_512_512 = self.modules.color(_illu_512_512)
        _colors_512 = T.Resize((h, w))(_colors_512_512)
        
        _hints_512_512 = self.modules.hints(_colors_512_512)
        
        _lineart_512_512 = self.modules.lineart(_illu_512_512)
        _lineart_512 = T.Resize((h, w))(_lineart_512_512)

        _mask_512_512 = self.modules.mask(_illu_512_512)
        _mask_512 = T.Resize((h, w))(_mask_512_512)

        illustration = self.to_tensor(_illu_512)
        lineart = self.to_tensor(_lineart_512)
        mask = self.to_tensor(_mask_512)

        lineart[lineart > 0.5] = 1.0
        lineart[lineart <= 0.5] = 0.0
        
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        
        hints_colors = self.to_tensor(T.Resize((h, w))(_hints_512_512.hints))
        hints_mask = self.to_tensor(T.Resize((h, w))(_hints_512_512.mask))
        hints = torch.cat([hints_colors, hints_mask], dim=0)

        masked_lineart = lineart * mask
        masked_illustration = illustration * (1 - mask)
        interpolation = masked_illustration + masked_lineart
        composition = torch.cat([interpolation, mask], dim=0)

        return Data(artist_id, composition, hints, illustration)

    @classmethod
    def from_config(cls, config: str, path: str) -> "ModularDataset":
        from paintstorch2.data import (
            COLOR_SIMPLIFIERS,
            HINTS_GENERATORS,
            LINEART_GENERATORS,
            MASK_GENERATORS,
        )
        
        with open(config, "r") as f:
            content = f.read()
        data = yaml.load(content, Loader=yaml.FullLoader)

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

        return cls(Modules(color, hints, lineart, mask), path)