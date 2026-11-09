"""Classification metrics, Purity and Rand Index."""
from __future__ import annotations

from collections import Counter
from itertools import combinations


def confusion_matrix(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true) | set(y_pred))
    pos = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[pos[t]][pos[p]] += 1
    return m, labels


def classification_report(y_true, y_pred, labels=None) -> dict:
    m, labels = confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    per_class = {}
    for i, c in enumerate(labels):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(labels))) - tp
        fn = sum(m[i]) - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": sum(m[i]), "tp": tp, "fp": fp, "fn": fn}
    acc = sum(m[i][i] for i in range(len(labels))) / n if n else 0.0
    macro = {k: sum(v[k] for v in per_class.values()) / len(labels) for k in ("precision", "recall", "f1")}
    weighted = {k: sum(v[k] * v["support"] for v in per_class.values()) / n for k in ("precision", "recall", "f1")}
    return {"accuracy": acc, "per_class": per_class, "macro": macro, "weighted": weighted,
            "confusion_matrix": m, "labels": labels}


def contingency(clusters, classes):
    """{cluster: Counter(class -> count)}"""
    table = {}
    for k, c in zip(clusters, classes):
        table.setdefault(k, Counter())[c] += 1
    return table


def purity(clusters, classes) -> float:
    """purity(Ω, C) = 1/N * Σ_k max_j |ω_k ∩ c_j|"""
    n = len(classes)
    table = contingency(clusters, classes)
    return sum(max(cnt.values()) for cnt in table.values()) / n if n else 0.0


def _c2(x: int) -> int:
    return x * (x - 1) // 2


def rand_index_counts(clusters, classes) -> dict:
    """Rand Index from the contingency table: TP = sum C(n_kj, 2), TP+FP = sum C(|w_k|, 2), TP+FN = sum C(|c_j|, 2)."""
    n = len(classes)
    table = contingency(clusters, classes)
    tp = sum(_c2(v) for cnt in table.values() for v in cnt.values())
    tp_fp = sum(_c2(sum(cnt.values())) for cnt in table.values())
    tp_fn = sum(_c2(v) for v in Counter(classes).values())
    fp = tp_fp - tp
    fn = tp_fn - tp
    total = _c2(n)
    tn = total - tp - fp - fn
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "pairs": total,
            "rand_index": (tp + tn) / total if total else 0.0}


def rand_index(clusters, classes) -> float:
    return rand_index_counts(clusters, classes)["rand_index"]


def rand_index_bruteforce(clusters, classes) -> float:
    """O(N^2) literal definition — used to verify the fast version on small samples."""
    agree = total = 0
    for i, j in combinations(range(len(classes)), 2):
        same_k = clusters[i] == clusters[j]
        same_c = classes[i] == classes[j]
        agree += int(same_k == same_c)
        total += 1
    return agree / total if total else 0.0


def majority_label_map(clusters, classes) -> dict:
    """cluster id -> majority true class (used to name clusters)."""
    return {k: cnt.most_common(1)[0][0] for k, cnt in contingency(clusters, classes).items()}
