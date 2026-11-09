"""Exact (MD5) and near-duplicate (TF-IDF cosine) detection."""
from __future__ import annotations

import hashlib
import re

import numpy as np


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def exact_duplicate_groups(texts: list[str]) -> list[list[int]]:
    buckets: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        buckets.setdefault(hashlib.md5(_norm(t).encode()).hexdigest(), []).append(i)
    return [g for g in buckets.values() if len(g) > 1]


def shingles(text: str, k: int = 5) -> set:
    w = re.findall(r"[a-z0-9]+", str(text).lower())
    return {" ".join(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def near_duplicate_pairs(tfidf_matrix: np.ndarray, threshold: float = 0.9, block: int = 512):
    """Return [(i, j, cosine)] for i<j with cosine >= threshold (rows already L2-normalised)."""
    X = tfidf_matrix.astype(np.float32)
    n = X.shape[0]
    pairs = []
    for s in range(0, n, block):
        S = X[s:s + block] @ X.T
        for r in range(S.shape[0]):
            i = s + r
            js = np.where(S[r, i + 1:] >= threshold)[0] + i + 1
            for j in js:
                pairs.append((i, int(j), float(S[r, j])))
    return pairs


def duplicate_report(texts, tfidf_matrix, threshold=0.9):
    exact = exact_duplicate_groups(texts)
    exact_pairs = {(min(a, b), max(a, b)) for g in exact for a in g for b in g if a < b}
    near = near_duplicate_pairs(tfidf_matrix, threshold)
    rows = []
    for i, j, cos in near:
        rows.append({"i": i, "j": j, "cosine": cos,
                     "jaccard_5shingle": jaccard(shingles(texts[i]), shingles(texts[j])),
                     "type": "exact" if (i, j) in exact_pairs else "near"})
    # keep the first doc of each duplicate group, drop the rest
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        a, b = find(r["i"]), find(r["j"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    for a, b in exact_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    to_drop = sorted(i for i in range(len(texts)) if find(i) != i)
    return {"exact_groups": exact, "pairs": rows, "drop_indices": to_drop}
