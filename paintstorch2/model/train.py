if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from typing import List, Union

    import paintstorch2.data.color as pt2_color
    import paintstorch2.data.dataset as pt2_dataset
    import paintstorch2.data.hints as pt2_hints
    import paintstorch2.data.lineart as pt2_lineart
    import paintstorch2.data.mask as pt2_mask
    import paintstorch2.model as pt2_model
    import paintstorch2.model.loss as pt2_loss
    import paintstorch2.model.network as pt2_net
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


    LATENT_DIM = 8
    CAPACITY = 16

    DATASET = "dataset"
    BATCH_SIZE = 2

    dataset = pt2_dataset.ModularPaintsTorch2Dataset(pt2_dataset.Modules(
        color=pt2_color.kMeansColorSimplifier((5, 15)),
        hints=pt2_hints.RandomHintsGenerator(),
        lineart=pt2_lineart.xDoGLineartGenerator(),
        mask=pt2_mask.kMeansMaskGenerator((2, 10)),
    ), DATASET, False)

    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)

    F1 = torch.jit.load(pt2_model.ILLUSTRATION2VEC)
    F2 = torch.jit.load(pt2_model.VGG16)
    
    S = pt2_net.Embedding(LATENT_DIM)
    G = pt2_net.Generator(LATENT_DIM, CAPACITY)
    D = pt2_net.Discriminator(CAPACITY)

    to_cuda(F1, F2, S, G, D)
    to_train(F1, F2, S, G, D)
    to_eval(F1, F2, S, G, D)

    for artist_id, composition, hints, style, illustration in loader:
        b, c, h, w = composition.size()

        artist_id = artist_id.cuda()
        composition = composition.cuda()
        hints = hints.cuda()
        style = style.cuda()
        illustration = illustration.cuda()
        noise = torch.rand((b, 1, h, w)).cuda()

        print("====== INPUTS")
        print("artist_id   ", tuple(artist_id.size()))
        print("composition ", tuple(composition.size()))
        print("hints       ", tuple(hints.size()))
        print("style       ", tuple(style.size()))
        print("illustration", tuple(illustration.size()))
        print("noise       ", tuple(noise.size()))

        features = F1(composition[:, :3])
        style = S(style)

        fake = G(composition, hints, features, style, noise)
        fake = composition[:, :3] + fake * composition[:, :-1]

        pred_fake = D(fake, features)
        pred_real = D(illustration, features)

        print("====== OUTPUTS")
        print("features    ", tuple(features.size()))
        print("style       ", tuple(style.size()))
        print("fake        ", tuple(fake.size()))
        print("pred_fake   ", tuple(pred_fake.size()))
        print("pred_real   ", tuple(pred_real.size()))

        break