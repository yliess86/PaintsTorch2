from evaluation.export import PaintsTorch2
from PIL import Image

import argparse
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as onnx


SIZE = 512, 512
H_SIZE = 128, 128


class TorchInference:
    def __init__(self, model: str, features: int, bn: bool = False) -> None:
        self.model = PaintsTorch2(features, model, "models/i2v.pth", bn)
        self.model = torch.jit.script(self.model).cuda()

    def __call__(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        y = self.model(torch.from_numpy(x).cuda(), torch.from_numpy(h).cuda())
        return y[0].cpu().numpy().transpose((1, 2, 0))


parser = argparse.ArgumentParser()
parser.add_argument("-x",         type=str, required=True)
parser.add_argument("-c",         type=str, required=False)
parser.add_argument("-m",         type=str, required=False)
parser.add_argument("--features", type=int, required=True)
parser.add_argument("--model",    type=str, required=True)
parser.add_argument("--bn",       action="store_true")
args = parser.parse_args()


model = TorchInference(args.model, args.features, args.bn)

m = np.ones((*SIZE, 1), dtype=np.float32)
if args.m is not None:
    m = Image.open(args.m).convert("L").resize(SIZE)
    m = np.array(m, dtype=np.float32) / 255
    m = m[:, :, None]

x = Image.open(args.x).convert("RGB").resize(SIZE)
x = np.array(x, dtype=np.float32) / 255
x = np.concatenate([x, m], axis=-1).transpose((2, 0, 1))
x = np.expand_dims(x, axis=0)
x[:, :3] = (x[:, :3] - 0.5) / 0.5

h = np.zeros((*H_SIZE, 4),dtype=np.float32)
if args.c is not None:
    h = Image.open(args.c).convert("RGBA").resize(H_SIZE)
    h = np.array(h, dtype=np.float32) / 255

h = h.transpose((2, 0, 1))
h = np.expand_dims(h, axis=0)
h[:, :3] = (h[:, :3] - 0.5) / 0.5

y = model(x, h) * 0.5 + 0.5
y = Image.fromarray((y * 255).astype(np.uint8))
y.show()