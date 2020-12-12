if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from typing import List, Union

    import importlib_resources as ir
    import paintstorch2.model.loss as pt2_loss
    import paintstorch2.model.network as pt2_net
    import paintstorch2.model.res as pt2_res
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    def to_cuda(*args: List[Union[nn.Module, torch.Tensor]]) -> None:
        for e in args:
            e.cuda()


    def to_train(*args: List[nn.Module]) -> None:
        for e in args:
            e.train()
            for param in e.parameters():
                param.requires_grad = True


    def to_eval(*args: List[nn.Module]) -> None:
        for e in args:
            e.eval()
            for param in e.parameters():
                param.requires_grad = False


    ILLUSTRATION2VEC = "illustration2vec.ts"
    VGG16 = "vgg16.ts"
    
    LATENT_DIM = 32
    CAPACITY = 16

    F1 = torch.jit.load(str(ir.files(pt2_res).joinpath(ILLUSTRATION2VEC)))
    F2 = torch.jit.load(str(ir.files(pt2_res).joinpath(VGG16)))
    
    S = pt2_net.Embedding(LATENT_DIM)
    G = pt2_net.Generator(LATENT_DIM, CAPACITY)
    D = pt2_net.Discriminator(CAPACITY)

    to_cuda(F1, F2, S, G, D)
    to_train(F1, F2, S, G, D)
    to_eval(F1, F2, S, G, D)