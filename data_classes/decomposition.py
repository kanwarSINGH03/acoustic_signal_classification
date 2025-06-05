from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
                raise NotImplementedError(
                    f"Feature extraction for '{feature}' is not implemented yet."
                )
            case "van":
                raise NotImplementedError(
                    f"Feature extraction for '{feature}' is not implemented yet."
                )

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
