import torch
from models.vanilla_gan import GenMLP, DiscMLP
from models.dcgan import GenDCGAN, DiscDCGAN

def test_vanilla_shapes():
    G, D = GenMLP(100), DiscMLP()
    z = torch.randn(8,100)
    x = G(z); y = D(x)
    assert x.shape == (8,1,28,28) and y.shape == (8,1)

def test_dcgan_shapes():
    G, D = GenDCGAN(100), DiscDCGAN()
    z = torch.randn(8,100)
    x = G(z); y = D(x)
    assert x.shape == (8,1,28,28) and y.shape == (8,1)
