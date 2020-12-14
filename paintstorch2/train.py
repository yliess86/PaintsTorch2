if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from typing import List, Union

    import argparse
    import multiprocessing
    import os
    import paintstorch2.data as pt2_data
    import paintstorch2.model as pt2_model
    import torch
    import torch.nn as nn


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


    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim",   type=int, default=128)
    parser.add_argument("--capacity",     type=int, default=64)
    parser.add_argument("--epochs",       type=int, default=200)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--dataset",      type=str, default="dataset")
    parser.add_argument("--checkpoints",  type=str, default="checkpoints")
    parser.add_argument("--tensorboards", type=str, default="tensorboards")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints, exist_ok=True)
    if not os.path.exists(args.tensorboards):
        os.makedirs(args.tensorboards, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tensorboards)

    α = 1e-4        # AdamW Learning Rate
    β = 0.5, 0.9    # AdamW Betas
    ε_drift = 1e-3  # Discriminator Drifiting
    λ1 = 1e-4       # Adversarial Loss Weight
    λ2 = 10         # Gradient Penalty Weight

    dataset = pt2_data.ModularPaintsTorch2Dataset(pt2_data.Modules(
        color=pt2_data.kMeansColorSimplifier((5, 15)),
        hints=pt2_data.RandomHintsGenerator(),
        lineart=pt2_data.xDoGLineartGenerator(),
        mask=pt2_data.kMeansMaskGenerator((2, 10)),
    ), args.dataset, False)

    n = multiprocessing.cpu_count()
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=n)

    F1 = torch.jit.load(pt2_model.ILLUSTRATION2VEC)
    F2 = torch.jit.load(pt2_model.VGG16)
    
    S = pt2_model.Embedding(args.latent_dim)
    G = pt2_model.Generator(args.latent_dim, args.capacity)
    D = pt2_model.Discriminator(args.capacity)

    GP = pt2_model.GradientPenalty(D, λ2)
    MSE = nn.MSELoss()

    to_cuda(F1, F2, S, G, D, GP, MSE)
    to_eval(F1, F2)

    GS_parameters = list(G.parameters()) + list(S.parameters())
    optim_GS = AdamW(GS_parameters, lr=α, betas=β)
    optim_D = AdamW(D.parameters(), lr=α, betas=β)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        total_𝓛_D = 0
        total_𝓛_G = 0

        pbar = tqdm(loader, desc="Batch")
        for i, batch in enumerate(pbar):
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
            
            𝓛_fake = D(fake, features).mean(0).view(1)
            𝓛_real = D(illustration, features).mean(0).view(1)
            𝓛_critic = 𝓛_fake - 𝓛_real
            𝓛_p = GP(illustration, fake, features) + ε_drift * (𝓛_real ** 2)

            𝓛_D = 𝓛_critic + 𝓛_p
            𝓛_D.backward()

            optim_D.step()
            total_𝓛_D += 𝓛_D.item() / len(loader)

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

            features1 = F2(fake)
            with torch.no_grad():
                features2 = F2(illustration)

            𝓛_adv = - D(fake, features).mean()
            𝓛_content = MSE(features1, features2)

            𝓛_G = 𝓛_content + λ1 * 𝓛_adv
            𝓛_G.backward()

            optim_GS.step()
            total_𝓛_G += 𝓛_G.item() / len(loader)

            # =============
            # BATCH LOGGING
            # =============
            pbar.set_postfix(𝓛_D=total_𝓛_D, 𝓛_G=total_𝓛_G)

        # =============
        # EPOCH LOGGING
        # =============
        writer.add_scalar("𝓛_D", total_𝓛_D, epoch)
        writer.add_scalar("𝓛_G", total_𝓛_G, epoch)

        to_eval(S, G, D)
        
        _, composition, hints, style, illustration = dataset[7]
        c, h, w = composition.size()

        composition = composition.unsqueeze(0).cuda()
        hints = hints.unsqueeze(0).cuda()
        style = style.unsqueeze(0).cuda()
        illustration = illustration.unsqueeze(0).cuda()
        noise = torch.rand((1, 1, h, w)).cuda()

        with torch.no_grad():
            features = F1(composition[:, :3])
            style_embedding = S(style)
            fake = G(composition, hints, features, style_embedding, noise)
            fake = composition[:, :3] + fake * composition[:, :-1]

        composition = composition.squeeze(0).cpu()
        hints = hints.squeeze(0).cpu()
        style = style.squeeze(0).cpu()
        illustration = illustration.squeeze(0).cpu()
        fake = fake.squeeze(0).cpu()

        writer.add_image("composition/color", composition[:3], epoch)
        writer.add_image("composition/mask", composition[None, -1], epoch)
        writer.add_image("hints/color", hints[:3], epoch)
        writer.add_image("hints/mask", hints[None, -1], epoch)
        writer.add_image("style", style, epoch)
        writer.add_image("illustration", illustration, epoch)

        torch.save({
            "args": vars(args),
            "S": S.state_dict(),
            "G": G.state_dict(),
            "D": D.state_dict(),
        }, os.path.join(
            args.checkpoints,
            f"paintstorch2_{epoch:0{len(str(args.epochs))}d}.pth",
        ))