import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    DCGAN Generator Architecture for 64x64 RGB Face Image Generation.
    Latent Vector (nz): 100
    Generator Features (ngf): 64
    Output Channels (nc): 3
    Output Resolution: 64x64
    """
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: Z (nz x 1 x 1) -> ConvTranspose2d -> (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # State: (ngf * 8) x 4 x 4 -> ConvTranspose2d -> (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # State: (ngf * 4) x 8 x 8 -> ConvTranspose2d -> (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # State: (ngf * 2) x 16 x 16 -> ConvTranspose2d -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # State: (ngf) x 32 x 32 -> ConvTranspose2d -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Final Output Shape: [batch_size, 3, 64, 64] with values in [-1, 1]
        )

    def forward(self, x):
        # Support both [B, nz] and [B, nz, 1, 1] input shapes
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return self.main(x)
