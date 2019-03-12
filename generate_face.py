import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


# Root directory for dataset
dataroot = "./celeba"

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# load and use trained parameters
# reference: https://qiita.com/sh-tatsuno/items/42fccff90c98103dffc9
param_path = "./param/netG_inter7914.pth"
param = torch.load(param_path, map_location='cpu')
# Create the generator
netG = Generator(ngpu).to(device)
netG.load_state_dict(param)

for i in range(20):
    # Generate batch of latent vectors
    # noise = torch.randn(batch_size, nz, 1, 1, device=device)
    noise = torch.randn(1, nz, 1, 1, device=device)
    # Generate fake image batch with G
    fake = netG(noise)

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("generated images")
    plt.imshow(np.transpose(vutils.make_grid(fake.detach(), padding=2, normalize=True), (1,2,0)))
    plt.pause(0.5)

"""
# Generate batch of latent vectors
# noise = torch.randn(batch_size, nz, 1, 1, device=device)
noise = torch.randn(1, nz, 1, 1, device=device)
# Generate fake image batch with G
fake = netG(noise)

for i in range(5):
    # visualization
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("generated images")
    plt.imshow(np.transpose(vutils.make_grid(fake.detach()[i], padding=2, normalize=True), (1,2,0)))
    plt.pause(1)
    # plt.close()
"""

