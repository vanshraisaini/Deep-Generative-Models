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

    def discriminator_loss(self, real, fake):
        real_scores = self.discriminator(real)
        fake_scores = self.discriminator(fake.detach())   # don't backprop into G here
        loss_real = F.relu(1.0 - real_scores).mean()
        loss_fake = F.relu(1.0 + fake_scores).mean()
        return loss_real + loss_fake, real_scores.mean(), fake_scores.mean()

    def generator_loss(self, fake):
        return -self.discriminator(fake).mean()

    def sample(self, num_samples):
        # returns images in [0, 1] so they can be plotted directly
        self.generator.eval()
        with torch.no_grad():
            images = self.generate(num_samples)
        self.generator.train()
        return (images + 1) / 2
