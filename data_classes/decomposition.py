from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, MiniBatchDictionaryLearning


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
        **kwargs: Optional[dict],
    ):
        self.X = df_X.reset_index(drop=True)  # keep indices aligned
        self.Y = df_Y.reset_index(drop=True)
        self.feature = feature

        self.kwargs = kwargs

        self.X_reduced = self.get_features(feature)

    def get_features(self, feature: str) -> np.ndarray:
        """
        Extract the specified feature from the DataFrame.
        """
        match feature:
            case "raw":
                # No feature extraction, just return the raw data
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

            case "autoencoder":

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

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_dim = self.kwargs["input_dim"]
                latent_dim = self.kwargs["latent_dim"]
                weight_decay = self.kwargs["weight_decay"]
                lr = self.kwargs["lr"]

                model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(
                    device
                )

                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                )
                n_epochs = self.kwargs["n_epochs"]
                batch_size = self.kwargs["batch_size"]
                X_np = self.X.values.astype(np.float32)
                X_norm = (X_np - X_np.min()) / (X_np.max() - X_np.min())
                X_tensor = torch.from_numpy(X_norm)
                dataset = TensorDataset(X_tensor, X_tensor)
                dataloader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, drop_last=True
                )

                for epoch in range(1, n_epochs + 1):
                    model.train()
                    running_loss = 0.0

                    for batch_idx, (batch_X, _) in enumerate(dataloader):
                        batch_X = batch_X.to(device)  # (16, 36000)
                        optimizer.zero_grad()

                        # Forward pass: encode → decode
                        reconstructed = model(batch_X)

                        # Compute loss
                        loss = criterion(reconstructed, batch_X)

                        # Backprop + optimize
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        if (batch_idx + 1) % 100 == 0:
                            avg_loss = running_loss / 100
                            print(
                                f"Epoch [{epoch}/{n_epochs}], "
                                f"Batch [{batch_idx+1}/{len(dataloader)}], "
                                f"Loss: {avg_loss:.6f}"
                            )
                            running_loss = 0.0

                model.eval()
                with torch.no_grad():
                    X_all = X_tensor.to(device)  # (3309, 36000)
                    latent_codes = model.encode(X_all)  # (3309, 128)
                    latent_codes_np = latent_codes.cpu().numpy()

                return latent_codes_np

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
        sample = torch.tensor(self.X_reduced[index], dtype=torch.float32)
        label = torch.tensor(self.Y.iloc[index], dtype=torch.long)
        return sample, label
