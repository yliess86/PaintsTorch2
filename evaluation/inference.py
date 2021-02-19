from PIL import Image
from paintstorch.network import Illustration2Vec, Generator

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument("-x",      type=str, required=True)
parser.add_argument("-c",      type=str, required=False)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--bn",    action="store_true")
args = parser.parse_args()

normalize = T.Normalize((0.5, ) * 3, (0.5, ) * 3)

fake = torch.zeros((1, 4, 512, 512))
F1 = Illustration2Vec("./models/i2v.pth").eval()
F1 = torch.jit.trace(F1, fake).cuda()

ckpt = torch.load(args.model)

G = nn.DataParallel(Generator(32, bn=args.bn))
G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)
G = G.module.eval().cuda()

x = Image.open(args.x).convert("RGB")
size = x.size

x = x.resize((512, 512))
x = np.array(x) / 255
x = np.concatenate([x, np.ones((*x.shape[:2], 1))], axis=-1)
x = torch.from_numpy(x).permute((2, 0, 1)).unsqueeze(0).cuda().float()
x[:, :3] = normalize(x[:, :3])

h = (
    np.random.rand(128, 128, 4) * 255 if args.c is None else
    Image.open(args.c).convert("RGBA").resize((128, 128))
)
h = np.array(h) / 255
h[:, :, -1] = np.sum(h[:, :, :3], axis=-1) != 0 if args.c else 0
h = torch.from_numpy(h).permute((2, 0, 1)).unsqueeze(0).cuda().float()
h[:, :3] = normalize(h[:, :3])

with torch.no_grad():
    mask = x[:, -1].unsqueeze(1)
    y, *_ = G(x, h, F1(x))
    y = x[:, :3] * (1 - mask) + y * mask

y = y.squeeze(0).permute((1, 2, 0)).cpu().numpy()
y = (y * 0.5) + 0.5
y = Image.fromarray((y * 255).astype(np.uint8))
y = y.resize(size)

y.show()
