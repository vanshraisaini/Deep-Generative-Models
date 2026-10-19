import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=512, device=device, dtype=dtype),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256*14*14, device=device, dtype=dtype),
            nn.LeakyReLU(inplace=True),  
        )

        self.deconv = nn.Sequential(
            self.up(256, 128),     # (B, 128, 28, 28)
            self.up(128, 64),      # (B, 64, 56, 56)
            self.up(64, 32),       # (B, 32, 112, 112)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, device=self.device, dtype=self.dtype), # (B, 3, 224, 224)
            nn.Sigmoid()
        )


    def forward(self, input):
        # input dim: (B, D)

        output = self.linear(input)                            #(B, 256*14*14)
        output = output.reshape(input.shape[0], 256, 14, 14)    #(B, 256, 14, 14)
        output = self.deconv(output)                           #(B, 3, 224, 224) 
        return output


    def up(self, i, o):
        return nn.Sequential(
            nn.ConvTranspose2d(i, o, kernel_size=4, stride=2, padding=1, device=self.device, dtype=self.dtype),
            nn.SiLU(),
        )


if(__name__ == "__main__"):

    decoder = Decoder(512, "cuda", torch.float32)

    input = torch.randn(10, 512, device="cuda", dtype=torch.float32)
    output = decoder(input)
    print(output.shape)

    