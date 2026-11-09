"""Content-based Top-K recommender (cosine similarity over TF-IDF)."""
from __future__ import annotations

import pandas as pd
import pytest

from textmining.preprocessing import PreprocessConfig, preprocess
from textmining.recommender import ContentRecommender, precision_at_k


@pytest.fixture(scope="module")
def rec(baseline_tfidf, dataset):
    return ContentRecommender(baseline_tfidf, dataset[1])


@pytest.mark.parametrize("doc", [0, 7, 1000, 2224])
@pytest.mark.parametrize("k", [1, 5, 20])
def test_similar_to_doc_contract(rec, doc, k):
    res = rec.similar_to_doc(doc, k)
    assert len(res) == k
    ids = [i for i, _ in res]
    scores = [s for _, s in res]
    assert doc not in ids                                   # self excluded
    assert len(set(ids)) == k
    assert scores == sorted(scores, reverse=True)           # descending
    assert all(-1e-6 <= s <= 1 + 1e-5 for s in scores)      # cosine of non-negative unit vectors


def test_scores_equal_cosine(rec):
    X = rec.X
    for i, s in rec.similar_to_doc(10, 5):
        assert s == pytest.approx(float(X[10] @ X[i]), abs=1e-6)


def test_dup_threshold_hides_near_copies(rec, texts):
    # BUS-007 'Jobs growth still slow in the US' has an exact copy (score 1.0)
    q = 6
    plain = rec.similar_to_doc(q, 5)
    assert plain[0][1] == pytest.approx(1.0, abs=1e-5)
    hidden = rec.similar_to_doc(q, 5, dup_threshold=0.9)
    assert len(hidden) == 5
    assert all(s < 0.9 for _, s in hidden)
    assert plain[0][0] not in [i for i, _ in hidden]


def test_free_text_query(rec):
    toks = preprocess("Chelsea striker scores twice as the club wins the Premiership football match", PreprocessConfig())
    res = rec.similar_to_text(toks, 5)
    assert len(res) == 5
    cats = rec.meta["category"].iloc[[i for i, _ in res]].tolist()
    assert cats.count("sport") >= 4
    assert [s for _, s in res] == sorted((s for _, s in res), reverse=True)


def test_empty_or_unknown_query_returns_empty(rec):
    assert rec.similar_to_text([], 5) == []
    assert rec.similar_to_text(["qwertyuiop", "zzzzzz"], 5) == []


def test_explain_terms(rec):
    q = 6
    res = rec.similar_to_doc(q, 1)
    why = rec.explain(rec.X[q], res[0][0], 5)
    assert why and all(w > 0 for _, w in why)
    assert [w for _, w in why] == sorted((w for _, w in why), reverse=True)
    assert all(t in rec.model.vocab for t, _ in why)


def test_small_corpus_k_larger_than_corpus():
    from textmining.tfidf import TfidfModel
    m = TfidfModel(min_df=1, max_df_ratio=1.0, max_features=None).fit([["a", "b"], ["a", "c"], ["b", "c"]])
    r = ContentRecommender(m, pd.DataFrame({"category": ["x", "y", "z"]}))
    res = r.similar_to_doc(0, 10)
    assert len(res) == 2 and 0 not in [i for i, _ in res]


def test_precision_at_k_real_data(rec, labels):
    sample = list(range(0, len(labels), 7))
    p5 = precision_at_k(rec, labels, 5, sample)
    assert p5 > 0.85
    p5_nodup = precision_at_k(rec, labels, 5, sample, dup_threshold=0.9)
    assert p5_nodup > 0.85


def test_precision_at_k_toy():
    from textmining.tfidf import TfidfModel
    m = TfidfModel(min_df=1, max_df_ratio=1.0, max_features=None).fit(
        [["goal", "match"], ["goal", "team"], ["bank", "rate"], ["bank", "share"]])
    r = ContentRecommender(m, pd.DataFrame({"category": ["s", "s", "b", "b"]}))
    assert precision_at_k(r, ["s", "s", "b", "b"], 1) == 1.0


def test_baseline_p_at_5(baseline_run):
    assert baseline_run["precision_at_5"] > 0.9
