import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Linear(4096, input_dim),
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


class Autoencoder_CNN(nn.Module):
    def __init__(self):
        super(Autoencoder_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=1),
            nn.MaxPool1d(4),
            nn.ReLU(True),
            nn.Conv1d(2, 5, kernel_size=3, padding=1),
            nn.MaxPool1d(4),
            nn.ReLU(True),
            nn.Conv1d(5, 10, kernel_size=3, padding=1),
            nn.MaxPool1d(4),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.ConvTranspose1d(10, 5, kernel_size=3, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.ConvTranspose1d(5, 3, kernel_size=3, padding=1, output_padding=0),
            nn.ReLU(True),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.ConvTranspose1d(3, 2, kernel_size=3, padding=1, output_padding=32),
            nn.ReLU(True),
            nn.ConvTranspose1d(2, 1, kernel_size=1),
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
