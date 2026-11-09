"""Multinomial Naive Bayes implemented from scratch (IIR ch. 13).

    P(c)        = N_c / N
    P(t | c)    = (T_ct + alpha) / (sum_t' T_ct' + alpha * |V|)       (Laplace / add-alpha)
    c_map(d)    = argmax_c  log P(c) + sum_{t in d} log P(t | c)

The vocabulary is built from the TRAINING documents only, so no test
information leaks into the model.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.log_unseen: dict[str, float] = {}
        self.vocab: set[str] = set()
        self.class_term_counts: dict[str, Counter] = {}
        self.class_doc_counts: dict[str, int] = {}

    def fit(self, token_lists: list[list[str]], labels: list[str]) -> "MultinomialNaiveBayes":
        n = len(labels)
        self.classes = sorted(set(labels))
        self.class_doc_counts = Counter(labels)
        self.class_term_counts = {c: Counter() for c in self.classes}
        for toks, y in zip(token_lists, labels):
            self.class_term_counts[y].update(toks)
        self.vocab = set()
        for c in self.classes:
            self.vocab.update(self.class_term_counts[c].keys())
        V = len(self.vocab)
        for c in self.classes:
            self.log_prior[c] = math.log(self.class_doc_counts[c] / n)
            total = sum(self.class_term_counts[c].values())
            denom = total + self.alpha * V
            self.log_likelihood[c] = {t: math.log((cnt + self.alpha) / denom)
                                      for t, cnt in self.class_term_counts[c].items()}
            self.log_unseen[c] = math.log(self.alpha / denom)  # term in V but never seen in c
        return self

    def class_scores(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(t for t in tokens if t in self.vocab)  # unknown terms ignored
        scores = {}
        for c in self.classes:
            ll = self.log_likelihood[c]
            unseen = self.log_unseen[c]
            s = self.log_prior[c]
            for t, k in counts.items():
                s += k * ll.get(t, unseen)
            scores[c] = s
        return scores

    def predict_one(self, tokens: list[str]) -> str:
        scores = self.class_scores(tokens)
        return max(scores, key=scores.get)

    def predict(self, token_lists: list[list[str]]) -> list[str]:
        return [self.predict_one(t) for t in token_lists]

    def predict_proba_one(self, tokens: list[str]) -> dict[str, float]:
        """Posterior via log-sum-exp normalisation."""
        s = self.class_scores(tokens)
        m = max(s.values())
        exp = {c: math.exp(v - m) for c, v in s.items()}
        z = sum(exp.values())
        return {c: v / z for c, v in exp.items()}

    def top_features(self, k: int = 15) -> dict[str, list[tuple[str, float]]]:
        """Most class-indicative terms: log P(t|c) - mean_c' log P(t|c')."""
        out = {}
        for c in self.classes:
            ratios = []
            for t, lp in self.log_likelihood[c].items():
                if self.class_term_counts[c][t] < 3:
                    continue
                others = [self.log_likelihood[o].get(t, self.log_unseen[o]) for o in self.classes if o != c]
                ratios.append((t, lp - sum(others) / len(others)))
            ratios.sort(key=lambda x: -x[1])
            out[c] = ratios[:k]
        return out


def stratified_split(labels: list[str], test_size: float = 0.2, seed: int = 42):
    """Manual stratified train/test split -> (train_idx, test_idx)."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    train, test = [], []
    for y in sorted(by_class):
        idx = by_class[y][:]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return sorted(train), sorted(test)
