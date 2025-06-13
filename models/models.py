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


class WaveNetClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(WaveNetClassifier, self).__init__()

        self.n_filters = 32
        self.filter_width = 3
        self.dilation_rates = [2**i for i in range(6)]  # [1, 2, 4, 8, 16, 32]

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_shape[0] if i == 0 else self.n_filters,
                    out_channels=self.n_filters,
                    kernel_size=self.filter_width,
                    dilation=d,
                    padding="same",
                )
                for i, d in enumerate(self.dilation_rates)
            ]
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(self.n_filters) for _ in self.dilation_rates]
        )

        self.final_conv = nn.Conv1d(
            in_channels=self.n_filters, out_channels=16, kernel_size=3, padding="same"
        )
        self.final_bn = nn.BatchNorm1d(16)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x shape: (batch, time, features) → transpose to (batch, features, time)
        x = x.permute(0, 2, 1)

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))

        x = F.relu(self.final_bn(self.final_conv(x)))
        x = self.global_max_pool(x).squeeze(-1)  # shape: (batch, 16)

        x = self.fc(x)
        return F.softmax(x, dim=1)
