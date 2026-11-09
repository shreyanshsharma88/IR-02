"""K-Means from scratch, logging every iteration."""
from __future__ import annotations

import numpy as np


def _sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
    xx = (X * X).sum(axis=1)[:, None]
    cc = (C * C).sum(axis=1)[None, :]
    d = xx - 2.0 * (X @ C.T) + cc
    np.maximum(d, 0, out=d)
    return d


class KMeans:
    def __init__(self, k: int = 5, init: str = "kmeans++", max_iter: int = 100, tol: float = 1e-4,
                 n_init: int = 5, seed: int = 42):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.sse_: float = float("inf")
        self.history: list[dict] = []
        self.init_indices: list[int] = []
        self.stop_reason: str = ""
        self.n_iter_: int = 0
        self.run_summary: list[dict] = []

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator):
        n = X.shape[0]
        if self.init == "random":
            idx = rng.choice(n, size=self.k, replace=False).tolist()
            return X[idx].copy(), idx
        idx = [int(rng.integers(n))]
        d2 = _sq_dists(X, X[idx]).min(axis=1)
        for _ in range(1, self.k):
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            nxt = int(rng.choice(n, p=probs))
            idx.append(nxt)
            d2 = np.minimum(d2, _sq_dists(X, X[[nxt]])[:, 0])
        return X[idx].copy(), idx

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        C, init_idx = self._init_centroids(X, rng)
        labels = np.full(X.shape[0], -1)
        history = []
        reason = f"max_iter ({self.max_iter}) reached"
        it = 0
        for it in range(1, self.max_iter + 1):
            D = _sq_dists(X, C)
            new_labels = D.argmin(axis=1)
            changed = int((new_labels != labels).sum())
            labels = new_labels
            newC = C.copy()
            for j in range(self.k):
                members = X[labels == j]
                if len(members):
                    newC[j] = members.mean(axis=0)
                else:  # empty cluster -> re-seed with farthest point
                    far = int(D.min(axis=1).argmax())
                    newC[j] = X[far]
            shift = float(np.sqrt(((newC - C) ** 2).sum(axis=1)).max())
            C = newC
            sse = float(_sq_dists(X, C)[np.arange(X.shape[0]), labels].sum())
            history.append({"iteration": it, "docs_changed": changed, "max_centroid_shift": round(shift, 6),
                            "SSE": round(sse, 4), "cluster_sizes": np.bincount(labels, minlength=self.k).tolist()})
            if changed == 0:
                reason = f"converged: no document changed cluster at iteration {it}"
                break
            if shift < self.tol:
                reason = f"converged: max centroid shift {shift:.2e} < tol {self.tol}"
                break
        sse = float(_sq_dists(X, C)[np.arange(X.shape[0]), labels].sum())
        return C, labels, sse, history, init_idx, reason, it

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        self.run_summary = []
        best = None
        for r in range(self.n_init):
            res = self._single_run(X, rng)
            self.run_summary.append({"run": r + 1, "SSE": round(res[2], 4), "iterations": res[6]})
            if best is None or res[2] < best[2]:
                best = res
        self.centroids, self.labels_, self.sse_, self.history, self.init_indices, self.stop_reason, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sq_dists(np.asarray(X, dtype=np.float32), self.centroids).argmin(axis=1)

    def top_terms(self, vocab: list[str], n: int = 10) -> dict[int, list[str]]:
        return {j: [vocab[i] for i in np.argsort(-self.centroids[j])[:n]] for j in range(self.k)}


def elbow(X: np.ndarray, ks, seed: int = 42, init: str = "kmeans++", n_init: int = 2):
    return [(k, KMeans(k=k, seed=seed, init=init, n_init=n_init, max_iter=50).fit(X).sse_) for k in ks]


def pca_2d(X: np.ndarray) -> np.ndarray:
    """2-D PCA projection, used only to plot the clusters."""
    Xc = X - X.mean(axis=0, keepdims=True)
    G = Xc @ Xc.T
    vals, vecs = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(vals)[::-1][:2]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))
