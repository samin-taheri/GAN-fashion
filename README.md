GANs on Fashion-MNIST (Vanilla GAN vs DCGAN)

Train and compare Vanilla GAN (MLP) and DCGAN (Conv) on Fashion-MNIST.
CPU-safe by default; runs anywhere.

Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

How to Train
Train Vanilla GAN
python3 train.py --model vanilla --epochs 5 --out runs/vanilla

Train DCGAN
python3 train.py --model dcgan --epochs 5 --out runs/dcgan

What Happens During Training

Loads Fashion-MNIST

Builds the selected model (Vanilla or DCGAN)

Trains generator + discriminator

Saves generated samples → samples/

Saves checkpoints → runs/<experiment>/ckpt/

Logs losses → runs/<experiment>/training_log.csv

Sample output filenames:

samples/vanilla_epoch1.png
samples/vanilla_epoch3.png
samples/vanilla_epoch5.png
samples/dcgan_epoch1.png
samples/dcgan_epoch3.png
samples/dcgan_epoch5.png

Project Structure
.
├── models/
│   ├── vanilla_gan.py
│   └── dcgan.py
│
├── utils/
│   └── viz.py
│
├── tests/
│   ├── test_shapes.py
│   └── test_train_step.py
│
├── samples/
├── runs/
├── train.py
├── requirements.txt
└── README.md

Model Architectures
Vanilla GAN (MLP)
class GenMLP(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 784), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(z.size(0), 1, 28, 28)

class DiscMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

DCGAN (Convolutional)
class GenDCGAN(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, ngf * 4 * 7 * 7)
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Tanh()
        )

class DiscDCGAN(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(ndf * 2 * 7 * 7, 1),
            nn.Sigmoid()
        )

Running Tests
pytest -q


Tests validate:

Generator output shape

Discriminator output shape

A full training step runs

Gradients update correctly

Example:

def test_dcgan_shapes():
    G, D = GenDCGAN(100), DiscDCGAN()
    z = torch.randn(8, 100)
    x = G(z); y = D(x)
    assert x.shape == (8, 1, 28, 28)
    assert y.shape == (8, 1)

Results
Vanilla GAN (Epoch Progress)
<img width="130" src="https://github.com/user-attachments/assets/f9805ea3-68b4-43e4-8b99-dbe459ae72ed"/> <img width="130" src="https://github.com/user-attachments/assets/f30435dd-082b-4b66-8183-755c9d033cdd"/> <img width="130" src="https://github.com/user-attachments/assets/cf176219-aa95-4923-8cb7-c598e9f72706"/>
DCGAN (Epoch Progress)
<img width="130" src="https://github.com/user-attachments/assets/df329e76-6249-4227-b0b6-e7e6a6ad62d0"/> <img width="130" src="https://github.com/user-attachments/assets/f2f9a89a-83f6-45b4-94a8-761394d1477e"/> <img width="130" src="https://github.com/user-attachments/assets/4da2c2b9-5520-443a-bbe5-f28000bcd232"/>
