if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm
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


    LATENT_DIM = 4
    CAPACITY = 16

    DATASET = "dataset"
    BATCH_SIZE = 2

    α = 1e-4        # AdamW Learning Rate
    β = 0.5, 0.9    # AdamW Betas
    ε_drift = 1e-3  # Discriminator Drifiting
    λ1 = 1e-4       # Adversarial Loss Weight
    λ2 = 10         # Gradient Penalty Weight

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

    GP = pt2_loss.GradientPenalty(D, λ2)
    MSE = nn.MSELoss()

    to_cuda(F1, F2, S, G, D, GP, MSE)
    to_eval(F1, F2)

    GS_parameters = list(G.parameters()) + list(S.parameters())
    optim_GS = AdamW(GS_parameters, lr=α, betas=β)
    optim_D = AdamW(D.parameters(), lr=α, betas=β)

    pbar = tqdm(loader, desc="Batch")
    for batch in pbar:
        artist_id, composition, hints, style, illustration = batch
        b, c, h, w = composition.size()

        artist_id = artist_id.cuda()
        composition = composition.cuda()
        hints = hints.cuda()
        style = style.cuda()
        illustration = illustration.cuda()
        noise = torch.rand((b, 1, h, w)).cuda()

        # =============
        # DISCRIMINATOR
        # =============
        pbar.set_description("Batch Discriminator")

        to_train(D)
        to_eval(S, G)
        optim_GS.zero_grad()
        optim_D.zero_grad()

        with torch.no_grad():
            features = F1(composition[:, :3])
            style_embedding = S(style)
            
            fake = G(composition, hints, features, style_embedding, noise)
            fake = composition[:, :3] + fake * composition[:, :-1]

        𝓛_D_fake = D(fake, features).mean(0).view(1)
        𝓛_D_fake.backward(retain_graph=True)
        
        𝓛_D_real = D(illustration, features).mean(0).view(1)
        𝓛_D_drift = ε_drift * (𝓛_D_real ** 2)
        𝓛_D = -1 * 𝓛_D_real + 𝓛_D_drift
        𝓛_D.backward(retain_graph=True)
        
        𝓛_GP = GP(illustration, fake, features)
        𝓛_GP.backward()

        optim_D.step()

        # =========
        # GENERATOR
        # =========
        pbar.set_description("Batch Generator")

        to_train(S, G)
        to_eval(D)
        optim_GS.zero_grad()
        optim_D.zero_grad()

        with torch.no_grad():
            features = F1(composition[:, :3])
        
        style_embedding = S(style)
        fake = G(composition, hints, features, style_embedding, noise)
        fake = composition[:, :3] + fake * composition[:, :-1]

        𝓛_D = D(fake, features).mean()
        𝓛_G = -λ1 * 𝓛_D
        𝓛_G.backward(retain_graph=True)

        features1 = F2(fake)
        with torch.no_grad():
            features2 = F2(illustration)

        𝓛_content = MSE(features1, features2)
        𝓛_content.backward()

        optim_GS.step()