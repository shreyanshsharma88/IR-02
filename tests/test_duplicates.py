"""Exact (MD5) and near-duplicate (TF-IDF cosine) detection."""
from __future__ import annotations

import pytest

from textmining.duplicates import (duplicate_report, exact_duplicate_groups, jaccard, near_duplicate_pairs,
                                   shingles)
from textmining.preprocessing import PreprocessConfig, preprocess_corpus
from textmining.tfidf import TfidfModel

TEXTS = [
    "Oil prices hit a record high as markets react to supply fears.",            # 0
    "OIL prices   hit a record high as markets\nreact to supply fears.",         # 1 exact dup of 0 (case/space)
    "The football club won the cup final after extra time in London.",          # 2
    "Oil prices hit a record high as markets react to supply fears today.",      # 3 near dup of 0
    "A new mobile phone with a camera and music player was launched.",           # 4
    "The football club won the cup final after extra time in London.",          # 5 exact dup of 2
    "Parliament debated the new immigration bill late into the night.",           # 6
]


def tfidf(texts):
    toks = preprocess_corpus(texts, PreprocessConfig())
    return TfidfModel(min_df=1, max_df_ratio=1.0, max_features=None).fit(toks).matrix


def test_exact_groups_ignore_case_and_whitespace():
    groups = exact_duplicate_groups(TEXTS)
    assert sorted(map(sorted, groups)) == [[0, 1], [2, 5]]


def test_no_exact_duplicates():
    assert exact_duplicate_groups(["a", "b", "c"]) == []


def test_shingles_and_jaccard():
    s = shingles("a b c d e f", k=5)
    assert s == {"a b c d e", "b c d e f"}
    assert shingles("Short Text", k=5) == {"short text"}   # fewer than k words -> one shingle
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0
    assert jaccard(shingles(TEXTS[0]), shingles(TEXTS[1])) == 1.0
    assert 0.5 < jaccard(shingles(TEXTS[0]), shingles(TEXTS[3])) < 1.0


def test_near_duplicate_pairs_found():
    X = tfidf(TEXTS)
    pairs = near_duplicate_pairs(X, 0.75)   # (0,3) cosine ~0.79
    found = {(i, j) for i, j, _ in pairs}
    assert {(0, 1), (0, 3), (1, 3), (2, 5)} <= found
    assert all(i < j and c >= 0.75 for i, j, c in pairs)
    assert (0, 4) not in found and (2, 6) not in found


def test_near_duplicate_blocking_same_result():
    X = tfidf(TEXTS)
    a, b = near_duplicate_pairs(X, 0.5, block=2), near_duplicate_pairs(X, 0.5, block=512)
    assert [(i, j) for i, j, _ in a] == [(i, j) for i, j, _ in b]
    assert [c for *_, c in a] == pytest.approx([c for *_, c in b], abs=1e-6)


def test_threshold_one_only_identical():
    X = tfidf(TEXTS)
    found = {(i, j) for i, j, _ in near_duplicate_pairs(X, 0.9999)}
    assert (0, 3) not in found and (2, 5) in found


def test_duplicate_report_drop_keeps_first_of_each_group():
    rep = duplicate_report(TEXTS, tfidf(TEXTS), 0.75)
    assert rep["drop_indices"] == [1, 3, 5]          # 0 and 2 survive as group representatives
    types = {(r["i"], r["j"]): r["type"] for r in rep["pairs"]}
    assert types[(0, 1)] == "exact" and types[(0, 3)] == "near" and types[(2, 5)] == "exact"
    for r in rep["pairs"]:
        assert 0 <= r["jaccard_5shingle"] <= 1


def test_exact_dups_dropped_even_below_threshold():
    """An exact text duplicate must be dropped even if its TF-IDF row is empty (cosine 0)."""
    texts = ["the and of", "THE and  of", "real content here"]
    rep = duplicate_report(texts, tfidf(texts), 0.9)
    assert rep["drop_indices"] == [1]


def test_real_dataset_exact_duplicates(texts):
    groups = exact_duplicate_groups(texts)
    assert len(groups) == 137
    assert all(len(g) == 2 for g in groups)          # 137 pairs
    assert sum(len(g) - 1 for g in groups) == 137


def test_real_dataset_drop_count(texts, baseline_tfidf):
    rep = duplicate_report(texts, baseline_tfidf.matrix, 0.9)
    assert len(rep["drop_indices"]) == 162
    assert len(rep["pairs"]) == 163
    assert len(texts) - len(rep["drop_indices"]) == 2063
