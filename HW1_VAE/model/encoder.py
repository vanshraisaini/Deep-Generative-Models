import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim, device, dtype):
        super().__init__()

        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype

        # Input Image Size: (B, 3, 224, 224)
        self.base_model = nn.Sequential(
            self.down(3, 32),     # (B, 32, 112, 112)
            self.down(32, 64),    # (B, 64, 56, 56)
            self.down(64, 128),   # (B, 128, 28, 28)
            self.down(128, 256),  # (B, 256, 14, 14)
            nn.Flatten(),         # (B, 256*14*14)
            nn.Linear(256 * 14 * 14, 512, device=device, dtype=dtype),
            nn.SiLU(),            # (B, 512)
        )

        self.mean_head = nn.Linear(512, latent_dim, device=device, dtype=dtype)
        self.log_var_head = nn.Linear(512, latent_dim, device=device, dtype=dtype)

    def forward(self, images):
        # images: (B, 3, 224, 224)
        h = self.base_model(images)            # (B, 512)
        mean = self.mean_head(h)               # (B, D)
        log_var = self.log_var_head(h)         # (B, D)
        log_var = torch.clamp(log_var, -10.0, 10.0)  # numerical safety for exp()
        return mean, log_var

    def down(self, i, o):
        # conv -> norm -> act, then a strided conv that halves the resolution
        return nn.Sequential(
            nn.Conv2d(i, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.GroupNorm(8, o, device=self.device, dtype=self.dtype),
            nn.SiLU(),
            nn.Conv2d(o, o, kernel_size=4, stride=2, padding=1, device=self.device, dtype=self.dtype),
            nn.GroupNorm(8, o, device=self.device, dtype=self.dtype),
            nn.SiLU(),
        )


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Encoder(latent_dim=128, device=dev, dtype=torch.float32)
    x = torch.randn((4, 3, 224, 224), device=dev)
    mean, log_var = encoder(x)
    print("Mean Shape:", mean.shape, "LogVar Shape:", log_var.shape)
    print("Params (M):", sum(p.numel() for p in encoder.parameters()) / 1e6)