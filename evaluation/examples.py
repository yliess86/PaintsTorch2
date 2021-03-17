from paintstorch.network import Generator, Illustration2Vec
from PIL import Image
from tqdm import tqdm

import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--files",     type=str, required=True)
parser.add_argument("--features",  type=int, default=32)
parser.add_argument("--model",     type=str, required=True)
parser.add_argument("--bn",        action="store_true")
args = parser.parse_args()

ckpt = torch.load(args.model)

G = nn.DataParallel(Generator(args.features, bn=args.bn))
G.load_state_dict(ckpt["G"] if "G" in ckpt.keys() else ckpt)
G = G.module.eval().cuda()

x = torch.zeros((2, 4, 512, 512))
F1 = torch.jit.trace(Illustration2Vec("./models/i2v.pth").eval(), x).cuda()

PC = torch.jit.load("./models/paintschainer.ts").cuda()

samples = [os.path.join(args.files, f) for f in os.listdir(args.files)]
pbar = tqdm(samples, desc="Generating")
with torch.no_grad():
    for sample in pbar:
        pbar.set_postfix(file=sample)

        # ==== TORCH
        x = Image.open(os.path.join(sample, "x.png"))
        h = Image.open(os.path.join(sample, "h.png"))
        m = Image.open(os.path.join(sample, "m.png"))

        x = np.array(x.convert("RGB").resize((512, 512))) / 255
        h = np.array(h.convert("RGBA").resize((512, 512))) / 255
        m = np.array(m.convert("L").resize((512, 512))) / 255

        x = np.concatenate([x, m[:, :, None]], axis=-1)

        x[:, :, :3] = (x[:, :, :3] - 0.5) / 0.5
        h[:, :, :3] = (h[:, :, :3] - 0.5) / 0.5

        x = torch.from_numpy(x).float().permute(2, 0, 1).cuda()
        h = torch.from_numpy(h).float().permute(2, 0, 1).cuda()
        h = F.interpolate(h.unsqueeze(0), size=(128, 128))

        y, *_ = G(x.unsqueeze(0), h, F1(x.unsqueeze(0)))
        y = y.squeeze(0).permute(1, 2, 0)
        y = y.cpu().detach().numpy()
        y = ((y * 0.5 + 0.5) * 255).astype(np.uint8)

        Image.fromarray(y).save(os.path.join(sample, "y_paintstorch2.png"))

        # ==== CHAINER
        x1 = cv2.imread(os.path.join(sample, "x.png"), cv2.IMREAD_GRAYSCALE)
        h = cv2.imread(os.path.join(sample, "h.png"), cv2.IMREAD_UNCHANGED)
        m = cv2.imread(os.path.join(sample, "m.png"), cv2.IMREAD_GRAYSCALE)
        
        x1 = np.asarray(x1, np.float32)
        x2 = x1.copy()
        x2 = cv2.resize(x2, (512, 512))
        x1 = cv2.resize(x1, (128, 128))
        m = cv2.resize(m, (512, 512))

        x1 = x1[:, :, None]
        x2 = x2[:, :, None]

        x1 = np.insert(x1, 1, -512, axis=2)
        x1 = np.insert(x1, 2, 128, axis=2)
        x1 = np.insert(x1, 3, 128, axis=2)

        h = cv2.resize(h, (128, 128))
        *h, a = cv2.split(h)
        h = cv2.merge(h)
            
        s = a != 0 
        x1[s, 1:] = cv2.cvtColor(h, cv2.COLOR_BGR2YUV)[s]

        x1 = torch.from_numpy(x1.transpose(2, 0, 1)).unsqueeze(0)
        x2 = torch.from_numpy(x2.transpose(2, 0, 1)).unsqueeze(0)

        y = PC(x1.cuda(), x2.cuda())[0].cpu().numpy()
        y = y.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
        y = cv2.cvtColor(y, cv2.COLOR_YUV2BGR)

        x = x2[0].cpu().numpy().transpose(1, 2, 0)
        m = (np.asarray(m, np.float32) / 255)[:, :, None]
        y = y * m + x * (1 - m)

        cv2.imwrite(os.path.join(sample, "y_paintschainer.png"), y)