if __name__ == "__main__":
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    from typing import List, Union

    import argparse
    import multiprocessing
    import numpy as np
    import os
    import paintstorch2.data as pt2_data
    import paintstorch2.metrics as pt2_metrics
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
    parser.add_argument("--exp",           type=str,  default="")
    parser.add_argument("--latent_dim",    type=int,  default=128)
    parser.add_argument("--capacity",      type=int,  default=64)
    parser.add_argument("--epochs",        type=int,  default=200)
    parser.add_argument("--batch_size",    type=int,  default=16)
    parser.add_argument("--preprocessed",  type=str,  default="preprocessed")
    parser.add_argument("--checkpoints",   type=str,  default="checkpoints")
    parser.add_argument("--tensorboards",  type=str,  default="tensorboards")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--amp",           action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints, exist_ok=True)
    if not os.path.exists(args.tensorboards):
        os.makedirs(args.tensorboards, exist_ok=True)

    # =====
    # SETUP
    # =====
    writer = SummaryWriter(log_dir=args.tensorboards)

    α = 1e-4                              # AdamW Learning Rate
    β = 0.5, 0.9                          # AdamW Betas
    γ = 0.1                               # Learning Rate Step Decay
    γ_step = int(0.75 * args.batch_size)  # Learning Rate Step Decay Step
    ε_drift = 1e-3                        # Discriminator Drifiting
    λ1 = 1e-4                             # Adversarial Loss Weight
    λ2 = 10                               # Gradient Penalty Weight

    dataset = pt2_data.PaintsTorch2Dataset(
        args.preprocessed, is_train=True,
    )

    batch_factor = torch.cuda.device_count() if args.data_parallel else 1
    loader = DataLoader(
        dataset,
        args.batch_size * batch_factor,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
    )

    if args.data_parallel:
        F1 = nn.DataParallel(torch.jit.load(pt2_model.ILLUSTRATION2VEC))
        F2 = nn.DataParallel(torch.jit.load(pt2_model.VGG16))
        
        S = nn.DataParallel(pt2_model.Embedding(args.latent_dim))
        G = nn.DataParallel(
            pt2_model.Generator(args.latent_dim, args.capacity)
        )
        D = nn.DataParallel(pt2_model.Discriminator(args.capacity))

        GP = nn.DataParallel(pt2_model.GradientPenalty(λ2))
        MSE = nn.DataParallel(nn.MSELoss())

        I3 = nn.DataParallel(pt2_metrics.InceptionV3Features())
    
    else:
        F1 = torch.jit.load(pt2_model.ILLUSTRATION2VEC)
        F2 = torch.jit.load(pt2_model.VGG16)
        
        S = pt2_model.Embedding(args.latent_dim)
        G = pt2_model.Generator(args.latent_dim, args.capacity)
        D = pt2_model.Discriminator(args.capacity)

        GP = pt2_model.GradientPenalty(λ2)
        MSE = nn.MSELoss()

        I3 = pt2_metrics.InceptionV3Features()

    to_cuda(F1, F2, S, G, D, GP, MSE, I3)
    to_eval(F1, F2, I3)

    GS_parameters = list(G.parameters()) + list(S.parameters())
    optim_GS = AdamW(GS_parameters, lr=α, betas=β)
    optim_D = AdamW(D.parameters(), lr=α, betas=β)

    scheduler_GS = StepLR(optim_GS, step_size=γ_step, gamma=γ)
    scheduler_D = StepLR(optim_D, step_size=γ_step, gamma=γ)
    
    scaler = GradScaler(enabled=args.amp)

    # ===============
    # VALIDATION DATA
    # ===============
    _, v_composition, v_hints, v_style, v_illust = dataset[7]
    c, h, w = v_composition.size()

    v_composition = v_composition.unsqueeze(0).cuda()
    v_hints = v_hints.unsqueeze(0).cuda()
    v_style = v_style.unsqueeze(0).cuda()
    v_illust = v_illust.unsqueeze(0).cuda()
    v_noise = torch.rand((1, 1, h, w)).cuda()

    with torch.no_grad():
        v_features = F1(v_composition[:, :3])
    
    # ========
    # TRAINING
    # ========
    e_pbar = tqdm(range(args.epochs), desc="Epoch")
    for epoch in e_pbar:
        total_𝓛_D = 0
        total_𝓛_G = 0

        fid_real_features = []
        fid_fake_features = []

        pbar = tqdm(loader, desc="Batch")
        for i, batch in enumerate(pbar):
            artist_id, composition, hints, style, illust = batch
            b, c, h, w = composition.size()

            artist_id = artist_id.cuda()
            composition = composition.cuda()
            hints = hints.cuda()
            style = style.cuda()
            illust = illust.cuda()
            noise = torch.rand((b, 1, h, w)).cuda()

            # ======
            # COMMON
            # ======
            to_train(D, S, G)
            optim_GS.zero_grad()
            optim_D.zero_grad()

            with autocast(enabled=args.amp):
                with torch.no_grad():
                    features = F1(composition[:, :3])
                
                style_embedding = S(style)
                fake = G(composition, hints, features, style_embedding, noise)
                fake = composition[:, :3] + fake * composition[:, :-1]
            
            # =============
            # DISCRIMINATOR
            # =============
            pbar.set_description("Batch Discriminator")
            
            d_fake = fake.detach()
            
            with autocast(enabled=args.amp):
                𝓛_fake = torch.relu(1 + D(d_fake, features).mean(0).view(1))
            scaler.scale(𝓛_fake).backward(retain_graph=True)

            with autocast(enabled=args.amp):
                𝓛_real = torch.relu(1 - D(illust, features).mean(0).view(1))
                𝓛_real_drift = -𝓛_real + ε_drift * (𝓛_real ** 2)
            scaler.scale(𝓛_real_drift).backward(retain_graph=True)

            𝓛_p = GP(D, illust, d_fake, features).mean(0)
            𝓛_p.backward()

            𝓛_D = 𝓛_fake + 𝓛_real_drift + 𝓛_p
            scaler.step(optim_D)
            total_𝓛_D += 𝓛_D.item() / len(loader)

            # =========
            # GENERATOR
            # =========
            pbar.set_description("Batch Generator")

            to_eval(D)
            optim_D.zero_grad()
            
            with autocast(enabled=args.amp):
                𝓛_adv = -λ1 * D(fake, features).mean(0)
            scaler.scale(𝓛_adv).backward(retain_graph=True)

            with autocast(enabled=args.amp):
                features1 = F2(fake)
                with torch.no_grad():
                    features2 = F2(illust)

            with autocast(enabled=args.amp):
                𝓛_content = MSE(features1, features2).mean(0)
            scaler.scale(𝓛_content).backward()

            𝓛_G = 𝓛_content + 𝓛_adv
            scaler.step(optim_GS)
            total_𝓛_G += 𝓛_G.item() / len(loader)

            # ==================
            # INCEPTION FEATURES
            # ==================
            pbar.set_description("Batch FID")

            with autocast(enabled=args.amp):
                with torch.no_grad():
                    fid_real_features.append(I3(illust).cpu().numpy())
                    fid_fake_features.append(I3(fake).cpu().numpy())

            # =============
            # BATCH LOGGING
            # =============
            pbar.set_postfix(𝓛_D=total_𝓛_D, 𝓛_G=total_𝓛_G)
            
            # ===========
            # BATCH RESET
            # ===========
            scaler.update()
            
        # ===========
        # EPOCH RESET
        # ===========
        scheduler_GS.step()
        scheduler_D.step()

        # ==========================
        # FRECHET INCEPTION DISTANCE
        # ==========================
        fid_real_features = np.concatenate(fid_real_features)
        fid_fake_features = np.concatenate(fid_fake_features)
        fid = pt2_metrics.fid(fid_real_features, fid_fake_features)

        # =============
        # EPOCH LOGGING
        # =============
        e_pbar.set_postfix(𝓛_D=total_𝓛_D, 𝓛_G=total_𝓛_G, FID=fid)

        # ==============
        # SCALAR LOGGING
        # ==============
        writer.add_scalar("𝓛_D", total_𝓛_D, epoch)
        writer.add_scalar("𝓛_G", total_𝓛_G, epoch)
        writer.add_scalar("FID", fid, epoch)

        # ==============
        # IMAGES LOGGING
        # ==============
        to_eval(S, G, D)

        with torch.no_grad():
            fake = v_composition[:, :3] + G(
                v_composition,
                v_hints,
                v_features,
                S(v_style),
                v_noise,
            ) * v_composition[:, :-1]

        composition = v_composition.squeeze(0).cpu()
        hints = v_hints.squeeze(0).cpu()
        style = v_style.squeeze(0).cpu()
        illust = v_illust.squeeze(0).cpu()
        fake = torch.clamp(fake.squeeze(0).cpu(), 0, 1)

        writer.add_image("composition/color", composition[:3], epoch)
        writer.add_image("composition/mask", composition[None, -1], epoch)
        writer.add_image("hints/color", hints[:3], epoch)
        writer.add_image("hints/mask", hints[None, -1], epoch)
        writer.add_image("style", style, epoch)
        writer.add_image("illustration", illust, epoch)
        writer.add_image("fake", fake, epoch)

        # ======
        # SAVING
        # ======
        torch.save({
            "args": vars(args),
            "S": S.state_dict(),
            "G": G.state_dict(),
            "D": D.state_dict(),
        }, os.path.join(
            args.checkpoints,
            f"paintstorch2_{args.exp}_{epoch:0{len(str(args.epochs))}d}.pth",
        ))