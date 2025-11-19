import torch.nn as nn

# generator
class GenMLP(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512),   nn.ReLU(True),
            nn.Linear(512, 784),   nn.Tanh()
        )
    def forward(self, z):
        return self.net(z).view(z.size(0), 1, 28, 28)

# discriminator
class DiscMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),   nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),     nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
