from __future__ import annotations

from typing import Union, Optional

import numpy as np
from scipy.linalg import svd


class PCA:

    def __init__(self, n_components:Union[float, int]):
        self._n_components:Union[float, int] = n_components
        self._components:Optional[np.ndarray] = None
        self._mean:Optional[np.ndarray] = None
        self._explained_variance:Optional[np.ndarray] = None

    def fit(self, X: np.ndarray)->PCA:
        self._mean:Optional[np.ndarray] = np.mean(X, axis=0)
        X_centered:np.ndarray = X - self._mean

        U, s, Vt = svd(X_centered, full_matrices=False)
        if isinstance(self._n_components, float) and 0 < self._n_components < 1:
            explained_variance:np.ndarray = (s ** 2) / (X.shape[0] - 1)
            total_variance:np.ndarray = np.sum(explained_variance)
            cumsum_variance:np.ndarray = np.cumsum(explained_variance) / total_variance
            n_components:int = np.argmax(cumsum_variance >= self._n_components) + 1
        else:
            n_components:int = self._n_components

        self._components:Optional[np.ndarray] = Vt[:n_components]
        self._explained_variance:Optional[np.ndarray] = (s[:n_components] ** 2) / (X.shape[0] - 1)

        return self

    def transform(self, X:np.ndarray):
        X_centered:np.ndarray = X - self._mean
        return X_centered @ self._components.T

    def inverse_transform(self, X_low:np.ndarray):
        X_high:np.ndarray = X_low @ self._components
        return X_high + self._mean

    def fit_transform(self, X:np.ndarray):
        return self.fit(X).transform(X)
