from onnx_tf.backend import prepare
from paintstorch.network import Generator, Illustration2Vec
from tensorflowjs.converters.tf_saved_model_conversion_v2 import (
    convert_tf_saved_model,
)

import argparse
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
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
        return x[:, :3] * (1 - mask) + y * mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--model",    type=str, required=True)
    parser.add_argument("--save",     type=str, required=True)
    parser.add_argument("--opset",    type=int, default=10)
    parser.add_argument("--bn",       action="store_true")
    args = parser.parse_args()


    ONNX = args.save
    TFPY = args.save.replace(".onnx", ".pb")
    TFJS = args.save.replace(args.save.split("/")[-1], "paintstorch2")


    model = PaintsTorch2(args.features, args.model, "./models/i2v.pth", args.bn)
    model = model.eval()

    x, h = torch.ones((1, 4, 512, 512)), torch.zeros((1, 4, 128, 128))
    x[:, :3] = (x[:, :3] - 0.5) / 0.5
    h[:, :3] = (h[:, :3] - 0.5) / 0.5
    y_torch = model(x, h)[0].numpy().transpose((1, 2, 0))

    parameters = (n for n, p in model.named_parameters())
    torch.onnx.export(
        model, (x, h), args.save,
        input_names=["input", "hints", *parameters],
        output_names=["illustration", ],
        do_constant_folding=True,
        export_params=True,
        opset_version=args.opset,
        verbose=True,
    )

    x, h = x.numpy(), h.numpy()
    session = rt.InferenceSession(ONNX)
    y_onnx = session.run(["illustration"], {"input": x, "hints": h})
    y_onnx = y_onnx[0][0].transpose((1, 2, 0))

    print()
    print("===== CONVERTION PyTorch > Onnx")
    print("Onnx close to Pytorch?", np.allclose(y_torch, y_onnx))
    print("Mean Abs Diff:", np.sum(np.abs(y_torch - y_onnx)) / (512 * 512 * 3))
    print()

    onnx_model = onnx.load(ONNX)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(args.save.replace(".onnx", ".pb"))

    tf_model = tf.saved_model.load(TFPY)
    y_tf = tf_model(input=x, hints=h)[0].numpy()
    y_tf = y_tf[0].transpose((1, 2, 0))

    print()
    print("===== CONVERTION Onnx > TensorFlow")
    print("Inputs:", tf_rep.inputs)
    print("Outputs:", tf_rep.outputs)
    print("Ops:", tf_rep.onnx_op_list)

    print("Signature:", tf_model.signatures["serving_default"])
    print("TensorFlow close to Pytorch?", np.allclose(y_torch, y_tf))
    print("Mean Abs Diff:", np.sum(np.abs(y_torch - y_tf)) / (512 * 512 * 3))
    print()

    convert_tf_saved_model(TFPY, TFJS)