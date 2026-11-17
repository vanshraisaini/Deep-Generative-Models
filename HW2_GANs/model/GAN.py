import torch
import torch.nn as nn
import torch.nn.functional as F

from model.generator import Generator
from model.discriminator import Discriminator


class GAN(nn.Module):

    def __init__(self, latent_dim, device, dtype):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype

        self.generator = Generator(latent_dim=latent_dim, device=device, dtype=dtype)
        self.discriminator = Discriminator(device=device, dtype=dtype)

    def generate(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim, device=self.device, dtype=self.dtype)
        return self.generator(z)                          # (B, 3, 128, 128) in [-1, 1]

    # Non-saturating GAN loss (softplus form of the original BCE loss)
    def discriminator_loss(self, real, fake):
        real_scores = self.discriminator(real)
        fake_scores = self.discriminator(fake.detach())   # don't backprop into G here
        loss = F.softplus(-real_scores).mean() + F.softplus(fake_scores).mean()
        return loss, real_scores.mean(), fake_scores.mean()

    def generator_loss(self, fake):
        return F.softplus(-self.discriminator(fake)).mean()

    def r1_penalty(self, real):
        # R1 (Mescheder et al. 2018, used in StyleGAN): penalize the gradient of D
        # on real images, so D stays smooth without needing spectral norm
        real = real.detach().requires_grad_(True)
        scores = self.discriminator(real)
        grad, = torch.autograd.grad(scores.sum(), real, create_graph=True)
        return grad.pow(2).sum(dim=(1, 2, 3)).mean()

    def sample(self, num_samples):
        # returns images in [0, 1] so they can be plotted directly
        self.generator.eval()
        with torch.no_grad():
            images = self.generate(num_samples)
        self.generator.train()
        return (images + 1) / 2
