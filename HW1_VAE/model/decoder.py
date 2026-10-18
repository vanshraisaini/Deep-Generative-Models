import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, latent_dim, device):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=56*56, device=device),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=56*56, out_features=20*56*56, device=device),
            nn.LeakyReLU(inplace=True),  

        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=20, out_channels=10, kernel_size=4, stride=2, padding=1, device=device   #(B, 10, 112, 112)
            ),  
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=10, out_channels=3, kernel_size=4, stride=2, padding=1, device=device    #(B, 3, 224, 224)
            ),
            nn.Sigmoid()
        )


    def forward(self, input):
        # input dim: (B, D)

        output = self.linear(input)                            #(B, 20*56*56)
        output = output.reshape(input.shape[0], 20, 56, 56)    #(B, 20, 56, 56)
        output = self.deconv(output)                           #(B, 3, 224, 224) 
        return output


if(__name__ == "__main__"):

    decoder = Decoder(512)

    input = torch.randn(10, 512)
    output = decoder(input)
    print(output.shape)

    