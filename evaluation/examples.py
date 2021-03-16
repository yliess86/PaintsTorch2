from PIL import Image

import argparse
import numpy as np
import os
import torch


parser = os.ArgumentParser()
parser.add_argument("--files", type=str, required=True)
args = parser.parse_arguments()

samples = [os.path.join(args.files, f) for f in os.listdir(args.files)]
for sample in samples:
    x = Image.open(os.path.join(sample, "x.png")).convert("RGBA").resize((512, 512))
    h = Image.open(os.path.join(sample, "h.png")).convert("RGBA").resize((128, 128))
    m = Image.open(os.path.join(sample, "m.png")).convert("L").resize((512, 512))

    x = np.array(x) / 255
    h = np.array(h) / 255
    m = np.array(m) / 255

    x = np.concatenate([x, m[:, :, None]], axis=-1)

    x[:, :, :3] = (x[:, :, :3] - 0.5) / 0.5
    h[:, :, :3] = (h[:, :, :3] - 0.5) / 0.5

    x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    h = torch.from_numpy(h).float().permute(2, 0, 1).unsqueeze(0)

    # y = model(x, h)
    # y = y.squeeze(0).permute(1, 2, 0)
    # y = (y * 0.5) + 0.5
    # y = y.cpu().detach().numpy()
    # y = (y * 255).astype(np.uint8)

    # Image.fromarray(y).save(os.path.join(sample, "y.png"))
