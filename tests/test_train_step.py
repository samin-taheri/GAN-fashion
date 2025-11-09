import torch, torch.nn as nn, torch.optim as optim
from models.vanilla_gan import GenMLP, DiscMLP

def test_one_train_step():
    G, D = GenMLP(100), DiscMLP()
    optG = optim.Adam(G.parameters(), lr=1e-4)
    optD = optim.Adam(D.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    b = 4
    real = torch.randn(b, 1, 28, 28)
    real_lbl = torch.ones(b, 1)
    fake_lbl = torch.zeros(b, 1)

    # D step
    z = torch.randn(b, 100)
    fake = G(z).detach()
    lossD = criterion(D(real), real_lbl) + criterion(D(fake), fake_lbl)
    optD.zero_grad(); lossD.backward(); optD.step()

    # G step
    z2 = torch.randn(b, 100)
    lossG = criterion(D(G(z2)), real_lbl)
    optG.zero_grad(); lossG.backward(); optG.step()

    assert lossD.item() >= 0 and lossG.item() >= 0
