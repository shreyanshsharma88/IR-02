"""2-D projection for VISUALISATION ONLY (clustering itself runs on full TF-IDF).
PCA via eigen-decomposition of the n x n Gram matrix of the centred data."""
import numpy as np


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    vals, vecs = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(vals)[::-1][:2]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
