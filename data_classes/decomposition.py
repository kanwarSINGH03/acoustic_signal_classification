import sys
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCA_features(Dataset):
    """
    PyTorch Dataset that holds PCA-reduced features.
    You must supply exactly one of `n_components` or `explained_variance`.
    """
    def __init__(
        self,
        df_X: pd.DataFrame,
        df_Y: pd.DataFrame,
        n_components: Optional[int] = None,
        explained_variance: Optional[float] = None,
    ):
        # exactly one of these two must be non-None
        if (n_components is None) == (explained_variance is None):
            raise ValueError("Please provide exactly one of `n_components` or `explained_variance`.")
        
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.Y = df_Y.reset_index(drop=True)            # keep labels aligned
        self._fit_pca_and_scale(df_X.reset_index(drop=True))

    def _fit_pca_and_scale(self, X: pd.DataFrame):
        # 1) Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2) Choose components by variance threshold or fixed count
        if self.explained_variance is not None:
            # find how many comps hit the target var
            full_pca = PCA(n_components=None)
            full_pca.fit(X_scaled)
            cumvar = np.cumsum(full_pca.explained_variance_ratio_)
            self.n_components = int(np.searchsorted(cumvar, self.explained_variance) + 1)

        # 3) Fit final PCA
        pca = PCA(n_components=self.n_components)
        self.X_reduced = pca.fit_transform(X_scaled)
        self.pca = pca   # store for later inspection

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_reduced[idx], dtype=torch.float32)
        y = torch.tensor(self.Y.iloc[idx], dtype=torch.long)
        return x, y