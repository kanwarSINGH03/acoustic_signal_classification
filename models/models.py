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
            nn.Conv1d(1, 6, kernel_size=3, padding=1),  # 1→6 channels
            nn.ReLU(True),
            nn.MaxPool1d(12),  # downsample by 12
            nn.Conv1d(6, 10, kernel_size=3, padding=1),  # 6→10 channels
            nn.ReLU(True),
            nn.MaxPool1d(12),  # downsample by 12 again
        )

        # Decoder: mirror of encoder with upsample and transposed-convs
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=12, mode="nearest"),  # upsample by 12
            nn.ConvTranspose1d(10, 6, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Upsample(
                scale_factor=12, mode="nearest"
            ),  # upsample back to original length
            nn.ConvTranspose1d(6, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
