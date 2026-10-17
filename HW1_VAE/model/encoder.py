import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # Input Image Size: (B, 3, 224, 224)
        self.base_model = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1),  #(B, 10, 224, 224)
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                #(B, 10, 112, 112)
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1), #(B, 20, 112, 112)
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                                #(B, 20, 56, 56)
            nn.Flatten(),                                                         #(B, 20*56*56)
        )

        self.mean_head = torch.nn.Sequential(
            nn.Linear(in_features=20*56*56, out_features=56*56),
            nn.SELU(),
            nn.Linear(in_features=56*56, out_features=latent_dim),
        )
        self.covariance_head = torch.nn.Sequential(
            nn.Linear(in_features=20*56*56, out_features=56*56),
            nn.SELU(),
            nn.Linear(in_features=56*56, out_features=latent_dim),
        )

    def forward(self, images):
        # Images shape: (B, 3, 224, 224)

        base_output = self.base_model(images)           #(B, 20*56*56))
        mean        = self.mean_head(base_output)       #(B, D)
        log_cov     = self.covariance_head(base_output) #(B, D)

        return mean, log_cov    


if __name__ == "__main__":
    encoder = Encoder(latent_dim = 10)
    
    noisy_images     = torch.randn((20, 3, 224, 224))
    mean, log_cov    = encoder(noisy_images)
    print("Mean Shape:", mean.shape)
    print("Covariance Shape:", log_cov.shape)
