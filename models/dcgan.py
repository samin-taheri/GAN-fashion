import torch.nn as nn

# 28x28 → project to 7x7 then upsample 7→14→28
class GenDCGAN(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, ngf * 4 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),  # 7->14
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),      # 14->28
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 7, 7)
        return self.main(x)

class DiscDCGAN(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),      # 28->14
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),  # 14->7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(ndf * 2 * 7 * 7, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
