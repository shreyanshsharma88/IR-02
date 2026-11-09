"""Evaluation metrics: IIR Figure 16.4 worked example (purity, Rand index) + classification report."""
from __future__ import annotations

import random

import pytest

from textmining.metrics import (classification_report, confusion_matrix, contingency, majority_label_map,
                                purity, rand_index, rand_index_bruteforce, rand_index_counts)

# IIR Fig. 16.4: cluster 1 = 5x + 1o ; cluster 2 = 1x + 4o + 1d ; cluster 3 = 2x + 3d   (N = 17)
CLUSTERS = [1] * 6 + [2] * 6 + [3] * 5
CLASSES = (["x"] * 5 + ["o"]) + (["x"] + ["o"] * 4 + ["d"]) + (["x"] * 2 + ["d"] * 3)


def test_iir_purity():
    assert purity(CLUSTERS, CLASSES) == pytest.approx((5 + 4 + 3) / 17)
    assert round(purity(CLUSTERS, CLASSES), 2) == 0.71


def test_iir_rand_index_counts():
    c = rand_index_counts(CLUSTERS, CLASSES)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (20, 20, 24, 72)
    assert c["pairs"] == 136
    assert c["rand_index"] == pytest.approx(92 / 136)
    assert round(rand_index(CLUSTERS, CLASSES), 2) == 0.68
    assert rand_index_bruteforce(CLUSTERS, CLASSES) == pytest.approx(92 / 136)


def test_contingency_and_majority():
    t = contingency(CLUSTERS, CLASSES)
    assert t[1] == {"x": 5, "o": 1} and t[2] == {"x": 1, "o": 4, "d": 1} and t[3] == {"x": 2, "d": 3}
    assert majority_label_map(CLUSTERS, CLASSES) == {1: "x", 2: "o", 3: "d"}


def test_perfect_and_degenerate_clusterings():
    y = ["a", "a", "b", "b"]
    assert purity([0, 0, 1, 1], y) == 1.0 and rand_index([7, 7, 3, 3], y) == 1.0
    assert purity([0, 0, 0, 0], y) == 0.5
    assert purity([0, 1, 2, 3], y) == 1.0            # singletons: purity is trivially 1 (IIR caveat)
    assert purity([], []) == 0.0 and rand_index([0], ["a"]) == 0.0


@pytest.mark.parametrize("seed", range(8))
def test_fast_rand_index_equals_bruteforce(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 120)
    k, j = rng.randint(1, 8), rng.randint(1, 6)
    cl = [rng.randrange(k) for _ in range(n)]
    cs = [rng.choice("abcdef"[:j]) for _ in range(n)]
    assert rand_index(cl, cs) == pytest.approx(rand_index_bruteforce(cl, cs), abs=1e-12)
    c = rand_index_counts(cl, cs)
    assert c["TP"] + c["FP"] + c["FN"] + c["TN"] == n * (n - 1) // 2


Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]


def test_confusion_matrix_rows_are_true_labels():
    m, labels = confusion_matrix(Y_TRUE, Y_PRED)
    assert labels == ["a", "b", "c"]
    assert m == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    m2, _ = confusion_matrix(["a"], ["b"], labels=["a", "b"])
    assert m2 == [[0, 1], [0, 0]]                      # row = actual 'a', column = predicted 'b'


def test_classification_report_hand_example():
    r = classification_report(Y_TRUE, Y_PRED)
    assert r["accuracy"] == pytest.approx(4 / 6)
    pc = r["per_class"]
    assert (pc["a"]["precision"], pc["a"]["recall"]) == pytest.approx((1.0, 2 / 3))
    assert pc["a"]["f1"] == pytest.approx(0.8)
    assert (pc["b"]["precision"], pc["b"]["recall"], pc["b"]["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert (pc["c"]["precision"], pc["c"]["recall"]) == pytest.approx((0.5, 1.0))
    assert pc["c"]["f1"] == pytest.approx(2 / 3)
    assert [pc[c]["support"] for c in "abc"] == [3, 2, 1]
    assert (pc["a"]["tp"], pc["a"]["fp"], pc["a"]["fn"]) == (2, 0, 1)
    assert r["macro"]["precision"] == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert r["macro"]["recall"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    assert r["macro"]["f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
    assert r["weighted"]["f1"] == pytest.approx((0.8 * 3 + 0.5 * 2 + 2 / 3 * 1) / 6)
    assert r["weighted"]["recall"] == pytest.approx(r["accuracy"])   # weighted recall == accuracy
    assert r["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_classification_report_zero_division():
    r = classification_report(["a", "a"], ["a", "b"])
    assert r["per_class"]["b"] == {**r["per_class"]["b"], "precision": 0.0, "recall": 0.0, "f1": 0.0}
