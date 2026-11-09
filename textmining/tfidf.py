"""Inverted index, TF-IDF (ltc + L2 normalisation) and cosine similarity."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class InvertedIndex:
    """term -> postings {doc_idx: raw term frequency}."""

    def __init__(self):
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_lengths: list[int] = []
        self.n_docs = 0

    def build(self, token_lists: list[list[str]]) -> "InvertedIndex":
        self.postings = defaultdict(dict)
        self.doc_lengths = []
        for d, toks in enumerate(token_lists):
            counts = Counter(toks)
            for term, c in counts.items():
                self.postings[term][d] = c
            self.doc_lengths.append(len(toks))
        self.n_docs = len(token_lists)
        return self

    def df(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def lookup(self, term: str) -> dict[int, int]:
        return self.postings.get(term, {})

    def __len__(self):
        return len(self.postings)


class TfidfModel:
    def __init__(self, min_df: int = 2, max_df_ratio: float = 0.9, max_features: int | None = 5000,
                 sublinear_tf: bool = True, smooth_idf: bool = False, normalize: bool = True):
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.normalize = normalize
        self.index: InvertedIndex | None = None
        self.vocab: list[str] = []
        self.term_to_col: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.matrix: np.ndarray | None = None
        self.pruned_stats: dict = {}

    def _tf_weight(self, tf: int) -> float:
        if tf <= 0:
            return 0.0
        return 1.0 + math.log10(tf) if self.sublinear_tf else float(tf)

    def _idf(self, df: int, n: int) -> float:
        if self.smooth_idf:
            return math.log10((1 + n) / (1 + df)) + 1.0
        return math.log10(n / df) if df else 0.0

    def fit(self, token_lists: list[list[str]]) -> "TfidfModel":
        self.index = InvertedIndex().build(token_lists)
        n = self.index.n_docs
        max_df = self.max_df_ratio * n
        candidates = [(t, len(p)) for t, p in self.index.postings.items()
                      if self.min_df <= len(p) <= max_df]
        removed_rare = sum(1 for p in self.index.postings.values() if len(p) < self.min_df)
        removed_common = sum(1 for p in self.index.postings.values() if len(p) > max_df)
        # keep the most frequent (by total corpus frequency) terms when capping
        if self.max_features and len(candidates) > self.max_features:
            cf = {t: sum(self.index.postings[t].values()) for t, _ in candidates}
            candidates.sort(key=lambda x: (-cf[x[0]], x[0]))
            candidates = candidates[: self.max_features]
        candidates.sort(key=lambda x: x[0])
        self.vocab = [t for t, _ in candidates]
        self.term_to_col = {t: i for i, t in enumerate(self.vocab)}
        self.idf = {t: self._idf(df, n) for t, df in candidates}
        self.pruned_stats = {"raw_terms": len(self.index), "removed_rare(<min_df)": removed_rare,
                             "removed_common(>max_df)": removed_common, "kept_terms": len(self.vocab)}

        mat = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for t, col in self.term_to_col.items():
            idf = self.idf[t]
            for d, tf in self.index.postings[t].items():
                mat[d, col] = self._tf_weight(tf) * idf
        if self.normalize:
            mat = self._l2_normalize(mat)
        self.matrix = mat
        return self

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        lengths = np.sqrt((mat * mat).sum(axis=1))
        lengths[lengths == 0] = 1.0
        return mat / lengths[:, None]

    def transform(self, token_lists: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(token_lists), len(self.vocab)), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            for term, tf in Counter(toks).items():
                col = self.term_to_col.get(term)
                if col is not None:
                    out[i, col] = self._tf_weight(tf) * self.idf[term]
        return self._l2_normalize(out) if self.normalize else out

    def top_terms(self, doc_idx: int, k: int = 10) -> list[tuple[str, float]]:
        row = self.matrix[doc_idx]
        idx = np.argsort(-row)[:k]
        return [(self.vocab[i], float(row[i])) for i in idx if row[i] > 0]

    def idf_table(self):
        return sorted(((t, self.index.df(t), self.idf[t]) for t in self.vocab), key=lambda x: -x[2])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """cos(a,b) = a.b / (|a||b|).  Works for vectors or row-matrices."""
    a = np.atleast_2d(a).astype(np.float32)
    b = np.atleast_2d(b).astype(np.float32)
    na = np.sqrt((a * a).sum(axis=1)); na[na == 0] = 1
    nb = np.sqrt((b * b).sum(axis=1)); nb[nb == 0] = 1
    return (a @ b.T) / np.outer(na, nb)


def cosine_pair_scalar(u: dict[str, float], v: dict[str, float]) -> float:
    """Pure-python sparse cosine for illustration / verification."""
    dot = sum(w * v.get(t, 0.0) for t, w in u.items())
    nu = math.sqrt(sum(w * w for w in u.values()))
    nv = math.sqrt(sum(w * w for w in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0
