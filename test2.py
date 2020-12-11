import importlib_resources as ir

import paintstorch2.model.res as pt2_res
import torch


vgg16 = torch.jit.load(str(ir.files(pt2_res).joinpath("vgg16.ts")))
illustration2vec = torch.jit.load(str(ir.files(pt2_res).joinpath("illustration2vec.ts")))


print(vgg16(torch.rand((1, 3, 512, 512))).size())
print(illustration2vec(torch.rand((1, 3, 512, 512))).size())