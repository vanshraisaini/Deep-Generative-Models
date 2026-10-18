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
            self.down(3, 32),   #(B, 32, 112, 112)
            self.down(32, 64),  #(B, 64, 56, 56)
            self.down(64, 128), #(B, 128, 28, 28)
            self.down(128, 256), #(B, 256, 14, 14)
            nn.Conv2d(256, 256, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.SiLU(),
            nn.Flatten(),        #(B, 256*14*14)
            nn.Linear(in_features=256*14*14, out_features=56*56, device=device, dtype=dtype), #(B, 56*56)
            nn.LeakyReLU(inplace=True)

        )

        self.mean_head = torch.nn.Sequential(
            nn.Linear(in_features=56*56, out_features=latent_dim, device=device, dtype=dtype),
        )
        self.covariance_head = torch.nn.Sequential(
            nn.Linear(in_features=56*56, out_features=latent_dim, device=device, dtype=dtype),
        )

    def forward(self, images):
        # Images shape: (B, 3, 224, 224)

        base_output = self.base_model(images)           #(B, 20*56*56))
        mean        = self.mean_head(base_output)       #(B, D)
        log_cov     = self.covariance_head(base_output) #(B, D)

        return mean, log_cov

    def down(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, kernel_size=3, padding=1, device=self.device, dtype=self.dtype),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )



if __name__ == "__main__":
    encoder = Encoder(latent_dim = 10, device = "cuda", dtype = torch.float32)
    
    noisy_images     = torch.randn((20, 3, 224, 224), device="cuda", dtype=torch.float32)
    mean, log_cov    = encoder(noisy_images)
    print("Mean Shape:", mean.shape)
    print("Covariance Shape:", log_cov.shape)
