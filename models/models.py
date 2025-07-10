import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import Optional


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
    def __init__(self, input_shape, num_classes=3):
        super(WaveNetClassifier, self).__init__()

        self.n_filters = 32
        self.filter_width = 3
        self.dilation_rates = [2**i for i in range(6)]  # [1, 2, 4, 8, 16, 32]
        in_channels = input_shape[1]  # 64

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else self.n_filters,
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
        # x shape: (batch, 400, 64) → transpose to (batch, 64, 400)
        x = x.transpose(1, 2)  # (batch, in_channels=64, time=400)

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))

        x = F.relu(self.final_bn(self.final_conv(x)))
        x = self.global_max_pool(x).squeeze(-1)  # shape: (batch, 16)

        x = self.fc(x)
        return F.softmax(x, dim=1)


# ------------------------------
# Fixed DWT Layer
# ------------------------------
class DWT_1D(nn.Module):
    def __init__(self, wavename="db1"):
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)
        self.register_buffer("low_filter", dec_lo.view(1, 1, -1))
        self.register_buffer("high_filter", dec_hi.view(1, 1, -1))

    def forward(self, x):
        L = F.conv1d(x, self.low_filter, stride=2)
        H = F.conv1d(x, self.high_filter, stride=2)
        return L, H


# sound -> DWT -> Low1 -> Low2
#           |              High2
#           | -> High1
# ------------------------------
# DWT Block with 2-Level Decomposition
# ------------------------------
class WaveletFirstBlock(nn.Module):
    def __init__(self, wavename="db1"):
        super().__init__()
        self.dwt1 = DWT_1D(wavename)
        self.dwt2 = DWT_1D(wavename)

    def forward(self, x):
        L1, H1 = self.dwt1(x)  # Level 1: Approximation and Detail
        L2, H2 = self.dwt2(L1)  # Level 2: Further Approximation and Detail
        return L2, H1, H2  # Average-2, Detail-1, Detail-2


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(2, 5, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=5),
            nn.Conv1d(5, 10, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=5),
        )

    def forward(self, x):
        return self.block(x)


class DeConvBlock(nn.Module):
    def __init__(self):
        super(DeConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.ConvTranspose1d(10, 5, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=5, mode="nearest"),
            nn.ConvTranspose1d(5, 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=5, mode="nearest"),
            nn.ConvTranspose1d(2, 1, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.ConvTranspose1d(1, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.block(x)


class DWTNet(nn.Module):
    def __init__(self, wavename="db1"):
        super(DWTNet, self).__init__()

        self.dwt_block = WaveletFirstBlock(wavename)

        self.encoder = nn.ModuleList([ConvBlock(), ConvBlock(), ConvBlock()])
        self.decoder = nn.ModuleList([DeConvBlock(), DeConvBlock(), DeConvBlock()])

    def forward(self, x):

        L2, H1, H2 = self.dwt_block(x)

        L2_encoded, H1_encoded, H2_encoded = self.encode(L2, H1, H2)

        L2_decoded, H1_decoded, H2_decoded = self.decode(
            L2_encoded, H1_encoded, H2_encoded
        )

        return L2_decoded, H1_decoded, H2_decoded

    def encode(self, L2, H1, H2):

        L2_encoded = self.encoder[0](L2)
        H1_encoded = self.encoder[1](H1)
        H2_encoded = self.encoder[2](H2)
        return L2_encoded, H1_encoded, H2_encoded

    def decode(self, L2_encoded, H1_encoded, H2_encoded):
        L2_decoded = self.decoder[0](L2_encoded)
        H1_decoded = self.decoder[1](H1_encoded)
        H2_decoded = self.decoder[2](H2_encoded)

        return L2_decoded, H1_decoded, H2_decoded


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_size=1,
        stride=1,
        res_net=True,
        pool_stride: Optional[int] = None,
        k1=40,
        k2=100,
    ):
        self.res_net = res_net
        super().__init__()
        pad = 6
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=k1, padding=pad, stride=stride
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=k2, padding=pad),
        )
        self.pool = (
            nn.MaxPool1d(pool_size, stride=pool_stride)
            if pool_size > 1
            else nn.Identity()
        )
        if self.res_net:
            if pool_size > 1 or in_channels != out_channels:
                layers = [
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
                ]
                if pool_size > 1:
                    layers.append(nn.MaxPool1d(pool_size, stride=pool_stride))
                self.skip = nn.Sequential(*layers)
            else:
                self.skip = nn.Identity()

    def forward(self, x):
        if self.res_net:
            out = self.pool(self.conv(x))
            skip = self.skip(x)

            # --- align time dims by cropping both to the shorter length ---
            L = min(out.size(-1), skip.size(-1))
            out = out[..., :L]
            skip = skip[..., :L]

            return F.relu(out + skip)
        else:
            out = self.pool(self.conv(x))
            return F.relu(out)


class Convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=20, padding=(10 - 1) // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(2, 5, pool_size=10, stride=2, res_net=True),
            ResidualBlock(5, 8, pool_size=10, stride=2, res_net=True),
            nn.Flatten(),
            nn.Linear(632, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # ensure channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.block(x)


class Convolution_p2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=10, padding=(10 - 1) // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(
                2, 3, pool_size=10, stride=2, res_net=True, pool_stride=7, k1=10, k2=10
            ),
            ResidualBlock(
                3, 4, pool_size=10, stride=2, res_net=True, pool_stride=7, k1=10, k2=10
            ),
            nn.Flatten(),
            nn.Linear(96, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 2),
        )

    def forward(self, x):
        # ensure channel dim
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.block(x)
