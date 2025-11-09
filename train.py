import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, argparse
from tqdm import tqdm
from models.vanilla_gan import GenMLP, DiscMLP
from models.dcgan import GenDCGAN, DiscDCGAN
from utils.viz import save_image_grid

def get_data(batch_size):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['vanilla','dcgan'], default='vanilla')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cpu')  # CPU-safe
    zdim = 100

    if args.model=='vanilla':
        G, D = GenMLP(zdim), DiscMLP()
    else:
        G, D = GenDCGAN(zdim), DiscDCGAN()

    G, D = G.to(device), D.to(device)

    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    loader = get_data(args.batch_size)
    fixed_z = torch.randn(16, zdim)

    for epoch in range(1, args.epochs+1):
        for real,_ in tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}'):
            real = real.to(device)
            b = real.size(0)
            real_lbl = torch.ones(b,1)
            fake_lbl = torch.zeros(b,1)

            # --- D step ---
            z = torch.randn(b, zdim)
            fake = G(z).detach()
            optD.zero_grad()
            lossD = loss_fn(D(real), real_lbl) + loss_fn(D(fake), fake_lbl)
            lossD.backward(); optD.step()

            # --- G step ---
            z = torch.randn(b, zdim)
            optG.zero_grad()
            lossG = loss_fn(D(G(z)), real_lbl)
            lossG.backward(); optG.step()

        with torch.no_grad():
            imgs = G(fixed_z)
            os.makedirs('samples', exist_ok=True)
            save_image_grid(imgs, f'samples/{args.model}_epoch{epoch}.png')

        print(f'Epoch {epoch}: lossD={lossD.item():.3f}, lossG={lossG.item():.3f}')

if __name__=='__main__':
    main()
