import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import Encoder
from model.decoder import Decoder


class VAE(nn.Module):

    def __init__(self, latent_dim, device):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim, device=device)
        self.decoder = Decoder(latent_dim=latent_dim, device=device)

    def forward(self, inputs):
        #input dim: (B, 3, 224, 224)
        
        mean, log_var = self.encoder(inputs)              # mean: (B, D); log_cov: (B, D)
        eps = torch.randn_like(log_var)                   # eps: (B, D)   
        z = mean + eps*torch.exp(0.5 * log_var)
        recon = self.decoder(z)
        return recon, mean, log_var

    def compute_loss(self, recon, target, mean, log_var):
        recon_loss = F.mse_loss(recon, target, reduction="sum") / target.shape[0]
        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / target.shape[0]
        return recon_loss, kl
        

