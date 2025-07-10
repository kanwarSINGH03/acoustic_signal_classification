from typing import Optional
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning
from models import models
from librosa.feature import rms, zero_crossing_rate
import pywt


class PCA_features(Dataset):
    """
    Class to hold PCA features.

    You must supply exactly one of `n_components` or `explained_variance`.
    """

    def __init__(
        self,
        df_X: pd.DataFrame = None,
        df_Y: pd.DataFrame = None,
        n_components: Optional[int] = None,
        explained_variance: Optional[float] = None,
    ):
        # Check that exactly one of n_components/explained_variance is provided
        if (n_components is None) == (explained_variance is None):
            raise ValueError(
                "Please provide exactly one of `n_components` or `explained_variance`."
            )

        self.n_components = n_components
        self.explained_variance = explained_variance
        self.X = df_X.reset_index(drop=True)  # keep indices aligned
        self.Y = df_Y.reset_index(drop=True)

        # This will both fit PCA + scaler, and set self.X_reduced & self.pca
        self.X_reduced = self.get_pca_features()

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.Y)

    def get_pca_features(self):
        """
        Standardize `self.X`, then apply PCA.
        If `explained_variance` was passed, we first fit a full PCA to find how many
        components meet that threshold, then refit a second PCA with that many components.
        Otherwise, we simply fit PCA(n_components=self.n_components).

        Returns:
            X_reduced: 2D numpy array of shape (n_samples, chosen_components)
        """
        # 1) Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X.values)

        # 2) If user asked for explained_variance, determine n_components now
        if (self.explained_variance is not None) and (self.n_components is None):
            # Fit a "full" PCA
            full_pca = PCA(n_components=None)
            full_pca.fit(X_scaled)
            cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

            # Find the smallest index where cumvar >= explained_variance
            chosen = np.argmax(cumulative_variance >= self.explained_variance) + 1
            self.n_components = int(chosen)

        # 3) Fit the final PCA with `self.n_components`
        pca_final = PCA(n_components=self.n_components)
        X_reduced = pca_final.fit_transform(X_scaled)
        self.pca = pca_final  # store the fitted PCA object for later inspection
        return X_reduced

    def get_samples(self):
        """
        Return the raw numpy array of PCA‐reduced features.
        """
        return self.X_reduced

    def get_labels(self):
        """
        Return the label array (as numpy).
        """
        return self.Y.values

    def __getitem__(self, index):
        """
        Return one (feature, label) pair as PyTorch tensors.
        """
        sample = torch.tensor(self.X_reduced[index], dtype=torch.float32)
        label = torch.tensor(self.Y.iloc[index], dtype=torch.long)
        return sample, label


class Extract_Features(Dataset):
    def __init__(
        self,
        df_X: pd.DataFrame,
        df_Y: pd.DataFrame,
        feature: str,
        transform=None,
        **kwargs: Optional[dict],
    ):
        self.X = df_X.reset_index(drop=True)  # keep indices aligned
        self.Y = df_Y.reset_index(drop=True)
        self.feature = feature
        self.transform = transform
        self.kwargs = kwargs

        self.X_reduced = self.get_features(feature)

    def get_features(self, feature: str) -> np.ndarray:
        """
        Extract the specified feature from the DataFrame.
        """
        match feature:
            case "raw":
                # No feature extraction, just return the raw data
                # scaler = StandardScaler()
                # X_scaled = scaler.fit_transform(self.X.values)
                # return X_scaled
                return self.X.values

            case "pca":
                if "n_components" in self.kwargs:
                    n_components = self.kwargs["n_components"]
                else:
                    n_components = None
                if "explained_variance" in self.kwargs:
                    explained_variance = self.kwargs["explained_variance"]
                else:
                    explained_variance = None
                if n_components is None and explained_variance is None:
                    raise ValueError(
                        "Please provide exactly one of `n_components` or `explained_variance`."
                    )
                if n_components is not None and explained_variance is not None:
                    raise ValueError(
                        "Please provide exactly one of `n_components` or `explained_variance`."
                    )
                pca_features = PCA_features(
                    self.X, self.Y, n_components, explained_variance
                )
                return pca_features.get_samples()

            case "dict_learning":
                mbdl = MiniBatchDictionaryLearning(
                    n_components=self.kwargs["n_components"],
                    alpha=self.kwargs["alpha"],
                    max_iter=self.kwargs["max_iter"],
                    random_state=self.kwargs["random_state"],
                    batch_size=self.kwargs["batch_size"],
                )
                mbdl.fit(self.X.values)
                return mbdl.transform(self.X.values)

            case "autoencoder_cnn":

                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )

                weight_decay = self.kwargs["weight_decay"]
                lr = self.kwargs["lr"]

                model = models.Autoencoder_CNN().to(device)

                criterion = nn.criterionLoss()
                optimizer = optim.SGD(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                num_epochs = self.kwargs["n_epochs"]
                batch_size = self.kwargs["batch_size"]
                X_np = self.X.values.astype(np.float32)
                X_norm = (X_np - X_np.min()) / (X_np.max() - X_np.min())
                X_tensor = torch.from_numpy(X_norm)

                train_dataset = TensorDataset(X_tensor[:3000], X_tensor[:3000])
                test_dataset = TensorDataset(X_tensor[3000:], X_tensor[3000:])

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                for epoch in range(num_epochs):
                    model.train()
                    epoch_loss = 0.0

                    for audios, _labels in train_loader:
                        audios = audios.unsqueeze(1).to(device)  # now (batch, 1, 36000)
                        outputs = model(audios)
                        loss = criterion(outputs, audios)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item() * audios.size(0)

                    avg_loss = epoch_loss / len(train_loader.dataset)
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {avg_loss:.6f}"
                    )

                model.eval()

                train_latents = []

                test_latents = []

                with torch.no_grad():
                    for audios, labels in train_loader:
                        audios = audios.unsqueeze(1).to(device)
                        z = model.encode(audios)
                        train_latents.append(z.cpu())

                    train_latents = torch.cat(train_latents, dim=0)

                    for audios, labels in test_loader:
                        audios = audios.unsqueeze(1).to(device)
                        z = model.encode(audios)
                        test_latents.append(z.cpu())

                    test_latents = torch.cat(
                        test_latents, dim=0
                    )  # shape: (N_test, latent_dim)

                    latents = np.concatenate(
                        (train_latents.numpy(), test_latents.numpy()), axis=0
                    )
                    return latents.reshape(latents.shape[0], -1)

            case "autoencoder_linear":

                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )

                input_dim = self.kwargs["input_dim"]
                latent_dim = self.kwargs["latent_dim"]
                weight_decay = self.kwargs["weight_decay"]
                lr = self.kwargs["lr"]

                model = models.Autoencoder(
                    input_dim=input_dim, latent_dim=latent_dim
                ).to(device)

                criterion = nn.criterionLoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                num_epochs = self.kwargs["n_epochs"]
                batch_size = self.kwargs["batch_size"]
                X_np = self.X.values.astype(np.float32)
                X_norm = (X_np - X_np.min()) / (X_np.max() - X_np.min())
                X_tensor = torch.from_numpy(X_norm)

                train_dataset = TensorDataset(X_tensor[:3000], X_tensor[:3000])
                test_dataset = TensorDataset(X_tensor[3000:], X_tensor[3000:])

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                for epoch in range(num_epochs):
                    model.train()
                    epoch_loss = 0.0

                    for audios, _labels in train_loader:
                        audios = audios.to(device)

                        outputs = model(audios)
                        loss = criterion(outputs, audios)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item() * audios.size(0)

                    avg_loss = epoch_loss / len(train_loader.dataset)
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {avg_loss:.6f}"
                    )

                model.eval()

                train_latents = []

                test_latents = []

                with torch.no_grad():
                    for audios, labels in train_loader:
                        audios = audios.to(device)
                        z = model.encode(audios)
                        train_latents.append(z.cpu())

                    train_latents = torch.cat(train_latents, dim=0)

                    for audios, labels in test_loader:
                        audios = audios.to(device)
                        z = model.encode(audios)
                        test_latents.append(z.cpu())

                    test_latents = torch.cat(
                        test_latents, dim=0
                    )  # shape: (N_test, latent_dim)

                    return np.concatenate(
                        (train_latents.numpy(), test_latents.numpy()), axis=0
                    )

            case "amplitude_envelope":
                frame_size = self.kwargs["frame_size"]
                hop_length = self.kwargs["hop_length"]

                windows = sliding_window_view(self.X, window_shape=frame_size, axis=1)

                windows = windows[:, ::hop_length, :]

                envelope = windows.max(axis=2)

                return envelope
            case "rms_energy":
                all_signals = []
                frame_size = self.kwargs["frame_size"]
                hop_length = self.kwargs["hop_length"]
                for signal in self.X.values:
                    rms_signal = rms(
                        y=signal, frame_length=frame_size, hop_length=hop_length
                    )[0]
                    all_signals.append(rms_signal)
                return np.array(all_signals)
            case "zero_crossing_rate":
                all_signals = []
                frame_size = self.kwargs["frame_size"]
                hop_length = self.kwargs["hop_length"]
                for signal in self.X.values:
                    zcr_signal = zero_crossing_rate(
                        y=signal, frame_length=frame_size, hop_length=hop_length
                    )[0]
                    all_signals.append(zcr_signal)
                return np.array(all_signals)
            case "wave_transform":
                scales = range(1, self.kwargs["scales"] + 1)
                wavelet = self.kwargs["wavelet"]

                cwt_results = []
                for i in range(len(self.X.values)):
                    signal = self.X.values[i]
                    cwt_matrix, freqs = pywt.cwt(signal, scales, wavelet)
                    cwt_results.append(cwt_matrix)
                    print(f"Processed signal {i+1}/{len(self.X.values)}", end="\r")
                cwt_results = np.array(cwt_results)

                def random_frame_sample(cwt_matrix, num_frames=100, seed=None):
                    if seed is not None:
                        np.random.seed(seed)
                    total_frames = cwt_matrix.shape[1]
                    indices = np.random.choice(
                        total_frames, size=num_frames, replace=False
                    )
                    sampled = cwt_matrix[:, indices]
                    return sampled

                # Apply on all CWTs
                num_frames = self.kwargs["num_frames"]
                sampled_cwts = np.array(
                    [
                        random_frame_sample(cwt, num_frames=num_frames)
                        for cwt in cwt_results
                    ]
                )  # shape: (3309, 63, num_frames)
                return sampled_cwts.transpose(0, 2, 1)  # shape: (3309, num_frames, 63)
            case "dwt_transform":
                wavelet = self.kwargs["wavelet"]
                level = self.kwargs["level"]
                dwt_coeff = pywt.wavedec(self.X.values, wavelet=wavelet, level=level)
                return dwt_coeff[0]  # Return the approximation coefficients
            case "dwt_net":
                device = torch.device(
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )

                weight_decay = self.kwargs["weight_decay"]
                lr = self.kwargs["lr"]

                model = models.DWTNet("db1").to(device)

                num_epochs = self.kwargs["n_epochs"]
                batch_size = self.kwargs["batch_size"]
                X_np = self.X.values.astype(np.float32)
                X_norm = (X_np - X_np.min()) / (X_np.max() - X_np.min())
                X_tensor = torch.from_numpy(X_norm)

                train_dataset = TensorDataset(X_tensor[:3000], X_tensor[:3000])
                test_dataset = TensorDataset(X_tensor[3000:], X_tensor[3000:])

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False
                )

                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=10
                )

                criterion = nn.MSELoss()

                for epoch in range(1, num_epochs + 1):
                    model.train()
                    running_loss = 0.0
                    for audio, _ in train_loader:
                        audio = audio.unsqueeze(1).to(device)  # shape (B, 1, 36000)
                        out_L2, out_H1, out_H2 = model(audio)
                        with torch.no_grad():
                            tgt_L2, tgt_H1, tgt_H2 = model.dwt_block(audio)
                        loss = (
                            criterion(out_L2, tgt_L2)
                            + criterion(out_H1, tgt_H1)
                            + criterion(out_H2, tgt_H2)
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item() * audio.size(0)
                    epoch_loss = running_loss / len(train_loader.dataset)
                    scheduler.step(epoch_loss)
                    print(f"[AE] Epoch {epoch:03d}  Loss: {epoch_loss:.6f}")

                model.eval()
                train_latents = []
                test_latents = []

                with torch.no_grad():
                    for audios, _ in train_loader:
                        audios = audios.unsqueeze(1).to(device)
                        L2, H1, H2 = model.dwt_block(audios)
                        z_L2, z_H1, z_H2 = model.encode(L2, H1, H2)
                        pool = nn.AdaptiveAvgPool1d(output_size=128)
                        z_L2 = pool(z_L2.cpu()).to(device)
                        z_H1 = pool(z_H1.cpu()).to(device)
                        z_H2 = pool(z_H2.cpu()).to(device)
                        z = torch.cat([z_L2, z_H1, z_H2], dim=1)
                        z = z.view(z.size(0), -1)
                        train_latents.append(z.cpu())

                    train_latents = torch.cat(train_latents, dim=0)

                    for audios, _ in test_loader:
                        audios = audios.unsqueeze(1).to(device)
                        L2, H1, H2 = model.dwt_block(audios)
                        z_L2, z_H1, z_H2 = model.encode(L2, H1, H2)
                        pool = nn.AdaptiveAvgPool1d(output_size=128)
                        z_L2 = pool(z_L2.cpu()).to(device)
                        z_H1 = pool(z_H1.cpu()).to(device)
                        z_H2 = pool(z_H2.cpu()).to(device)
                        z = torch.cat([z_L2, z_H1, z_H2], dim=1)
                        z = z.view(z.size(0), -1)
                        test_latents.append(z.cpu())

                    test_latents = torch.cat(test_latents, dim=0)

                    latents = np.concatenate(
                        (train_latents.numpy(), test_latents.numpy()), axis=0
                    )
                    return latents.reshape(latents.shape[0], -1)

    def get_samples(self) -> np.ndarray:
        """
        Return the raw numpy array of features.
        """
        return self.X_reduced

    def get_labels(self) -> np.ndarray:
        """
        Return the label array (as numpy).
        """
        return self.Y.values

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.Y)

    def __getitem__(self, index):
        """
        Return one (feature, label) pair as PyTorch tensors.
        """
        sample = torch.tensor(self.get_samples()[index], dtype=torch.float32)
        label = torch.tensor(self.get_labels()[index], dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)

        return sample, label
