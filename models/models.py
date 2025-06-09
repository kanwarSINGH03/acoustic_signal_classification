import torch.nn as nn
import torch


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12000),
            nn.ReLU(True),
            nn.Linear(12000, 3000),
            nn.ReLU(True),
            nn.Linear(3000, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3000),
            nn.ReLU(True),
            nn.Linear(3000, 12000),
            nn.ReLU(True),
            nn.Linear(12000, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
