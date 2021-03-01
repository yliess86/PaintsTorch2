from paintstorch.network import Generator, Illustration2Vec

import argparse
import numpy as np
import onnxruntime as onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaintsTorch2(nn.Module):
    def __init__(
        self, features: int, g: str, f1: str, bn: bool = False,
    ) -> None:
        super(PaintsTorch2, self).__init__()
        
        ckpt = torch.load(g)
        G = nn.DataParallel(Generator(features, bn=bn))
        G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)

        self.F1 = Illustration2Vec(f1).eval().cpu()
        self.G = G.module.eval().cpu()

        for param in self.G.parameters():
            param.requires_grad = False
        for param in self.F1.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        mask = x[:, -1].unsqueeze(1)
        y, *_ = self.G(x, h, self.F1(x))
        y = x[:, :3] * (1 - mask) + y * mask
        return y


parser = argparse.ArgumentParser()
parser.add_argument("--features", type=int, default=32)
parser.add_argument("--model",    type=str, required=True)
parser.add_argument("--save",     type=str, required=True)
parser.add_argument("--bn",       action="store_true")
args = parser.parse_args()


model = PaintsTorch2(args.features, args.model, "./models/i2v.pth", args.bn)
model = model.eval()

x, h = torch.randn((2, 4, 512, 512)), torch.randn((2, 4, 128, 128))
x[:, :3] = (x[:, :3] - 0.5) / 0.5
h[:, :3] = (h[:, :3] - 0.5) / 0.5
y_torch = model(x, h).numpy()

torch.onnx.export(
    model, (x, h), args.save,
    input_names=["input", "hints", *(n for n, p in model.named_parameters())],
    output_names=["illustration"],
    dynamic_axes={
        "input"       : { 0: "batch" },
        "hints"       : { 0: "batch" },
        "illustration": { 0: "batch" },
    },
    do_constant_folding=True,
    export_params=True,
    opset_version=11,
    verbose=False,
)

x, h = x.numpy(), h.numpy()
session = onnx.InferenceSession(args.save)
y_onnx = session.run(["illustration"], {"input": x, "hints": h})[0]

print("Pytorch == Onnx ?", np.allclose(y_torch, y_onnx))