import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.linear = nn.Linear(latent_dim, 512 * 4 * 4, device=device, dtype=dtype)

        self.deconv = nn.Sequential(
            nn.BatchNorm2d(512, device=device, dtype=dtype),
            nn.ReLU(),
            self.up(512, 256),   # (B, 256, 8, 8)
            self.up(256, 128),   # (B, 128, 16, 16)
            self.up(128, 64),    # (B, 64, 32, 32)
            self.up(64, 32),     # (B, 32, 64, 64)
            self.up(32, 16),     # (B, 16, 128, 128)
            nn.Conv2d(16, 3, kernel_size=3, padding=1, device=device, dtype=dtype),  # (B, 3, 128, 128)
            nn.Tanh(),           # images in [-1, 1]
        )

    def forward(self, z):
        # z: (B, D)
        output = self.linear(z)                          # (B, 512*4*4)
        output = output.reshape(z.shape[0], 512, 4, 4)   # (B, 512, 4, 4)
        output = self.deconv(output)                     # (B, 3, 128, 128)
        return output

    def up(self, i, o):
        # same block as the HW1 decoder: nearest upsample + two 3x3 convs
        # (BatchNorm + ReLU is the standard DCGAN choice for the generator)
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(i, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.BatchNorm2d(o, device=self.device, dtype=self.dtype),
            nn.ReLU(),
            nn.Conv2d(o, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.BatchNorm2d(o, device=self.device, dtype=self.dtype),
            nn.ReLU(),
        )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(latent_dim=128, device=dev, dtype=torch.float32)
    z = torch.randn(4, 128, device=dev)
    with torch.no_grad():
        out = generator(z)
    print(out.shape, float(out.min()), float(out.max()))
    print("Params (M):", sum(p.numel() for p in generator.parameters()) / 1e6)
