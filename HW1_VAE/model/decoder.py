import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 512, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(512, 256 * 14 * 14, device=device, dtype=dtype),
            nn.SiLU(),
        )

        self.deconv = nn.Sequential(
            self.up(256, 128),   # (B, 128, 28, 28)
            self.up(128, 64),    # (B, 64, 56, 56)
            self.up(64, 32),     # (B, 32, 112, 112)
            self.up(32, 16),     # (B, 16, 224, 224)
            nn.Conv2d(16, 3, kernel_size=3, padding=1, device=device, dtype=dtype),  # (B, 3, 224, 224)
            nn.Sigmoid(),
        )

    def forward(self, input):
        # input: (B, D)
        output = self.linear(input)                           # (B, 256*14*14)
        output = output.reshape(input.shape[0], 256, 14, 14)  # (B, 256, 14, 14)
        output = self.deconv(output)                          # (B, 3, 224, 224)
        return output

    def up(self, i, o):
        # nearest upsample + two 3x3 convs: avoids checkerboard artifacts and
        # gives each scale refinement capacity
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(i, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.GroupNorm(8, o, device=self.device, dtype=self.dtype),
            nn.SiLU(),
            nn.Conv2d(o, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.GroupNorm(8, o, device=self.device, dtype=self.dtype),
            nn.SiLU(),
        )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = Decoder(128, dev, torch.float32)
    z = torch.randn(4, 128, device=dev)
    out = decoder(z)
    print(out.shape, float(out.min()), float(out.max()))
    print("Params (M):", sum(p.numel() for p in decoder.parameters()) / 1e6)