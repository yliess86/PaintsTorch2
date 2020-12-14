from paintstorch2.model.loss import GradientPenalty
from paintstorch2.model.network import Discriminator, Embedding, Generator

import importlib_resources as ir
import paintstorch2.model.res as pt2_res


ILLUSTRATION2VEC = str(ir.files(pt2_res).joinpath("illustration2vec.ts"))
VGG16 = str(ir.files(pt2_res).joinpath("vgg16.ts"))