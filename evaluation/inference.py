from PIL import Image
from paintstorch.network import Illustration2Vec, Generator

import argparse
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as onnx


class TorchInference:
    def __init__(self, model: str, features: int, bn: bool = False) -> None:
        fake = torch.zeros((1, 4, 512, 512))
        F1 = Illustration2Vec("./models/i2v.pth").eval()

        ckpt = torch.load(model)
        G = nn.DataParallel(Generator(features, bn=bn))
        G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)

        self.F1 = torch.jit.trace(F1, fake).cuda()
        self.G = G.module.eval().cuda()
        self.normalize = lambda x: (x - 0.5) / 0.5

    def __call__(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x).permute((2, 0, 1)).unsqueeze(0).cuda()
        x[:, :3] = self.normalize(x[:, :3])

        h = torch.from_numpy(h).permute((2, 0, 1)).unsqueeze(0).cuda()
        h[:, :3] = self.normalize(h[:, :3])

        with torch.no_grad():
            mask = x[:, -1].unsqueeze(1)
            y, *_ = self.G(x, h, self.F1(x))
            y = x[:, :3] * (1 - mask) + y * mask
        y = y.squeeze(0).permute((1, 2, 0)).cpu().numpy()

        y = (y * 0.5) + 0.5
        y = (y * 255).astype(np.uint8)
        return y


class OnnxInference:
    def __init__(self, model: str) -> None:
        self.sess = onnx.InferenceSession(model)
        self.normalize = lambda x: (x - 0.5) / 0.5

    def __call__(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        x = np.transpose(x, axes=(2, 0, 1))
        x[:, :3] = self.normalize(x[:, :3])
        x = np.expand_dims(x, axis=0)

        h = np.transpose(h, axes=(2, 0, 1))
        h[:, :3] = self.normalize(h[:, :3])
        h = np.expand_dims(h, axis=0)

        y = self.sess.run(["illustration"], { "input": x, "hints": h })[0][0]
        y = np.transpose(y, axes=(1, 2, 0))
        y = (y * 0.5) + 0.5
        y = (y * 255).astype(np.uint8)
        return y


parser = argparse.ArgumentParser()
parser.add_argument("-x",         type=str, required=True)
parser.add_argument("-c",         type=str, required=False)
parser.add_argument("--features", type=int, required=True)
parser.add_argument("--model",    type=str, required=True)
parser.add_argument("--bn",       action="store_true")
args = parser.parse_args()


is_onnx = args.model.endswith(".onnx")
model = (
    OnnxInference(args.model) if is_onnx else
    TorchInference(args.model, args.features, args.bn)
)

x = Image.open(args.x).convert("RGB")
size = x.size

x = np.array(x.resize((512, 512))) / 255
x = np.concatenate([x, np.ones((*x.shape[:2], 1))], axis=-1)

h = np.array(
    np.zeros((128, 128, 4)) * 255 if args.c is None else
    Image.open(args.c).convert("RGBA").resize((128, 128))
) / 255
h[:, :, -1] = np.sum(h[:, :, :3], axis=-1) != 0 if args.c else 0

y = model(x.astype(np.float32), h.astype(np.float32))
y = Image.fromarray(y).resize(size)
y.show()
