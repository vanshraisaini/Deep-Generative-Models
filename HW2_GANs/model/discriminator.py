import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Input Image Size: (B, 3, 128, 128)
        self.base_model = nn.Sequential(
            self.down(3, 32),     # (B, 32, 64, 64)
            self.down(32, 64),    # (B, 64, 32, 32)
            self.down(64, 128),   # (B, 128, 16, 16)
            self.down(128, 256),  # (B, 256, 8, 8)
            self.down(256, 512),  # (B, 512, 4, 4)
            nn.Flatten(),         # (B, 512*4*4)
            spectral_norm(nn.Linear(512 * 4 * 4, 1, device=device, dtype=dtype)),  # (B, 1)
        )

    def forward(self, images):
        # images: (B, 3, 128, 128) in [-1, 1]
        return self.base_model(images).squeeze(1)  # (B,) raw scores, no sigmoid

    def down(self, i, o):
        # spectral norm on every layer keeps D smooth (Lipschitz), which is the
        # main thing that stops GAN training from blowing up. No BatchNorm in D.
        return nn.Sequential(
            spectral_norm(nn.Conv2d(i, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(o, o, kernel_size=4, stride=2, padding=1, device=self.device, dtype=self.dtype)),
            nn.LeakyReLU(0.2),
        )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = Discriminator(device=dev, dtype=torch.float32)
    x = torch.randn((4, 3, 128, 128), device=dev)
    print("Score Shape:", discriminator(x).shape)
    print("Params (M):", sum(p.numel() for p in discriminator.parameters()) / 1e6)
