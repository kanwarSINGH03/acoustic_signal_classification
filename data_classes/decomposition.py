import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional
import numpy as np
import sys
import logging
import torch
from torch.utils.data import Dataset

class PCA_features(Dataset):
    """
    Class to hold PCA features.
    """
    def __init__(
        self,
        df_X: pd.DataFrame = None,
        df_Y: pd.DataFrame = None,
        n_components: Optional[int] = None,
        explained_variance: Optional[float] = None,
    ):
        self.n_components = n_components
        self.explained_variance = explained_variance
        self.X = df_X
        self.Y = df_Y

        self.X_reduced = self.get_pca_features()
        

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.X)
    
    def get_pca_features(self):
        """
        Get the features.
        """
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

        if self.explained_variance is not None and self.n_components is None:
            pca = PCA(n_components=None)
            pca.fit(self.X)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= self.explained_variance) + 1

            pca_final = PCA(n_components=n_components)
            X_reduced = pca_final.fit_transform(self.X)

            return X_reduced
        elif self.n_components is not None and self.explained_variance is None:
            pca = PCA(n_components=self.n_components)
            X_reduced = pca.fit_transform(self.X)

            return X_reduced
        else:
            logging.error("No PCA features were created. Please provide either n_components or explained_variance.")
            sys.exit(1)
            return None
        
    def get_samples(self):
        return self.X_reduced
    
    def get_labels(self):
        return self.Y.values
        
    def __getitem__(self, index):
        sample = torch.tensor(self.X_reduced[index], dtype=torch.float32)
        label = torch.tensor(self.Y.iloc[index], dtype=torch.long)
        return sample, label