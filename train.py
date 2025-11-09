import os, csv, argparse, random, numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.vanilla_gan import GenMLP, DiscMLP
from models.dcgan import GenDCGAN, DiscDCGAN
from utils.viz import save_image_grid

# -----------------------
# Reproducibility
# -----------------------
def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Data
# -----------------------
def get_loader(batch_size: int) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # map to [-1, 1]
    ])
    ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

# -----------------------
# Train
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vanilla", "dcgan"], default="vanilla")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--latent_dim", type=int, default=100)
    ap.add_argument("--lrG", type=float, default=2e-4)
    ap.add_argument("--lrD", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="runs/exp")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Folders
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "ckpt"), exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Seed & device
    set_seed(args.seed)
    device = torch.device("cpu")  # CPU-safe

    # Build models
    if args.model == "vanilla":
        G, D = GenMLP(args.latent_dim).to(device), DiscMLP().to(device)
    else:
        G, D = GenDCGAN(args.latent_dim).to(device), DiscDCGAN().to(device)

    optG = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    criterion = nn.BCELoss()

    loader = get_loader(args.batch_size)
    fixed_z = torch.randn(16, args.latent_dim, device=device)

    # CSV logging
    log_path = os.path.join(args.out, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lossD", "lossG"])

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for real, _ in pbar:
            real = real.to(device)
            b = real.size(0)
            real_lbl = torch.ones(b, 1, device=device)
            fake_lbl = torch.zeros(b, 1, device=device)

            # ---- Train D ----
            z = torch.randn(b, args.latent_dim, device=device)
            fake = G(z).detach()
            optD.zero_grad()
            loss_real = criterion(D(real), real_lbl)
            loss_fake = criterion(D(fake), fake_lbl)
            lossD = loss_real + loss_fake
            lossD.backward(); optD.step()

            # ---- Train G ----
            z = torch.randn(b, args.latent_dim, device=device)
            optG.zero_grad()
            pred = D(G(z))
            lossG = criterion(pred, real_lbl)
            lossG.backward(); optG.step()

            pbar.set_postfix(lossD=float(lossD), lossG=float(lossG))

        # Samples
        G.eval()
        with torch.no_grad():
            samples = G(fixed_z).cpu()
        save_image_grid(samples, f"samples/{args.model}_epoch{epoch}.png", nrow=4)

        # Checkpoint
        state = {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict(),
            "args": vars(args),
        }
        torch.save(state, os.path.join(args.out, "ckpt", f"ckpt_{epoch:03d}.pt"))

        # Log row
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, float(lossD.item()), float(lossG.item())])

        print(f"Epoch {epoch}: lossD={lossD.item():.3f}, lossG={lossG.item():.3f}")

    print("Done. Outputs ->", args.out)

if __name__ == "__main__":
    main()
