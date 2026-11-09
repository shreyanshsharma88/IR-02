"""Content-based Top-K document recommender (vector-space model).

score(q, d) = cos(v_q, v_d) = v_q . v_d   (both L2-normalised TF-IDF vectors)
Optionally skips near-duplicates of the query so the analyst sees *new* information.
"""
from __future__ import annotations

import numpy as np


class ContentRecommender:
    def __init__(self, tfidf_model, metadata):
        self.model = tfidf_model
        self.X = tfidf_model.matrix
        self.meta = metadata.reset_index(drop=True)

    def _rank(self, qvec: np.ndarray, k: int, exclude: set, dup_threshold: float | None):
        scores = self.X @ qvec.astype(np.float32)
        order = np.argsort(-scores)
        out = []
        for i in order:
            i = int(i)
            if i in exclude:
                continue
            if dup_threshold is not None and scores[i] >= dup_threshold:
                continue
            out.append((i, float(scores[i])))
            if len(out) == k:
                break
        return out

    def similar_to_doc(self, doc_idx: int, k: int = 5, dup_threshold: float | None = None):
        return self._rank(self.X[doc_idx], k, {doc_idx}, dup_threshold)

    def similar_to_text(self, tokens: list[str], k: int = 5):
        q = self.model.transform([tokens])[0]
        if not q.any():
            return []
        return self._rank(q, k, set(), None)

    def explain(self, a: np.ndarray, doc_idx: int, n: int = 8):
        """Terms contributing most to the dot product."""
        contrib = a * self.X[doc_idx]
        idx = np.argsort(-contrib)[:n]
        return [(self.model.vocab[i], float(contrib[i])) for i in idx if contrib[i] > 0]


def precision_at_k(rec: ContentRecommender, labels: list[str], k: int = 5, sample: list[int] | None = None,
                   dup_threshold: float | None = None):
    """Proxy relevance: a recommendation is 'relevant' if it shares the query's category."""
    idxs = sample if sample is not None else range(len(labels))
    precs = []
    for i in idxs:
        recs = rec.similar_to_doc(i, k, dup_threshold)
        if recs:
            precs.append(sum(labels[j] == labels[i] for j, _ in recs) / len(recs))
    return float(np.mean(precs)) if precs else 0.0
