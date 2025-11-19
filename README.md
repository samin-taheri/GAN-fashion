GANs on Fashion-MNIST (Vanilla GAN vs DCGAN)

This project implements and compares two Generative Adversarial Network architectures:

Vanilla GAN — simple MLP-based generator & discriminator

DCGAN — convolution-based generator & discriminator designed for image generation

The goal is to evaluate training stability, visual quality, and architecture impact on Fashion-MNIST.

All implementations run fully on CPU, making the project portable and easy to test on any machine.

Setup:

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt

How to Train the Models:
Train Vanilla GAN
python3 train.py --model vanilla --epochs 5 --out runs/vanilla

Train DCGAN
python3 train.py --model dcgan --epochs 5 --out runs/dcgan

What Happens During Training

Running the script will:

Load Fashion-MNIST

Build the selected model (Vanilla or DCGAN)

Train both generator and discriminator

Save generated images at each epoch → samples/

Save model checkpoints → runs/<experiment>/ckpt/

Log losses to → runs/<experiment>/training_log.csv

Samples are saved automatically as:

samples/vanilla_epoch1.png
samples/vanilla_epoch2.png
samples/dcgan_epoch1.png
samples/dcgan_epoch2.png
...

Project Structure:
```
.
├── models/
│   ├── vanilla_gan.py      # MLP generator & discriminator
│   └── dcgan.py            # Convolutional generator & discriminator
│
├── utils/
│   └── viz.py              # save_image_grid() helper
│
├── tests/
│   ├── test_shapes.py      # shape validation tests
│   └── test_train_step.py  # one training step test
│
├── samples/                # saved images after each epoch
├── runs/                   # logs & checkpoints
├── train.py                # main training script
├── requirements.txt
└── README.md
```

Model Architectures:
Vanilla GAN (MLP)
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

class DiscMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),   nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),     nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

DCGAN (Convolutional Generator + Discriminator)
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

Running the Tests:

To ensure correctness, run:

pytest -q

✔ Tests check:

Generator output shape = (batch, 1, 28, 28)

Discriminator output shape = (batch, 1)

A full training step runs without errors

Gradients and updates work correctly

Example shape test:

def test_dcgan_shapes():
    G, D = GenDCGAN(100), DiscDCGAN()
    z = torch.randn(8, 100)
    x = G(z); y = D(x)
    assert x.shape == (8, 1, 28, 28)
    assert y.shape == (8, 1)

Results:

Images and loss logs are saved automatically in:

samples/
runs/<experiment>/

<img width="122" height="122" alt="vanilla_epoch1 19 08 55" src="https://github.com/user-attachments/assets/f9805ea3-68b4-43e4-8b99-dbe459ae72ed" />
<img width="122" height="122" alt="vanilla_epoch3 19 08 55" src="https://github.com/user-attachments/assets/f30435dd-082b-4b66-8183-755c9d033cdd" />
<img width="122" height="122" alt="vanilla_epoch5 19 08 55" src="https://github.com/user-attachments/assets/cf176219-aa95-4923-8cb7-c598e9f72706" />
Vanilla Epochs:

<img width="122" height="122" alt="dcgan_epoch1" src="https://github.com/user-attachments/assets/df329e76-6249-4227-b0b6-e7e6a6ad62d0" />
<img width="122" height="122" alt="dcgan_epoch3" src="https://github.com/user-attachments/assets/f2f9a89a-83f6-45b4-94a8-761394d1477e" />
<img width="122" height="122" alt="dcgan_epoch5" src="https://github.com/user-attachments/assets/4da2c2b9-5520-443a-bbe5-f28000bcd232" />
DCGAN Epochs:




