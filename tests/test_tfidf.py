"""Unit tests for the hand-built inverted index, TF-IDF (SMART 'ltc') and cosine similarity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from textmining.tfidf import InvertedIndex, TfidfModel, cosine_pair_scalar, cosine_similarity

# tiny corpus, computed by hand below
DOCS = [
    ["a", "a", "b"],          # d0
    ["b", "c"],               # d1
    ["a", "d", "d", "d"],     # d2
]
N = 3


def full_model(**kw):
    """No pruning so every term survives."""
    params = dict(min_df=1, max_df_ratio=1.0, max_features=None)
    params.update(kw)
    return TfidfModel(**params).fit(DOCS)


# ------------------------------------------------------------------ inverted index
def test_inverted_index_postings_exact():
    ix = InvertedIndex().build(DOCS)
    assert dict(ix.postings) == {"a": {0: 2, 2: 1}, "b": {0: 1, 1: 1}, "c": {1: 1}, "d": {2: 3}}
    assert ix.n_docs == 3
    assert ix.doc_lengths == [3, 2, 4]
    assert len(ix) == 4


def test_inverted_index_df_and_lookup():
    ix = InvertedIndex().build(DOCS)
    assert {t: ix.df(t) for t in "abcd"} == {"a": 2, "b": 2, "c": 1, "d": 1}
    assert ix.df("zzz") == 0 and ix.lookup("zzz") == {}
    assert ix.lookup("d") == {2: 3}


# ------------------------------------------------------------------ weights
def test_vocab_sorted_and_idf_formula():
    m = full_model()
    assert m.vocab == ["a", "b", "c", "d"]
    assert m.idf["a"] == pytest.approx(math.log10(3 / 2))
    assert m.idf["b"] == pytest.approx(math.log10(3 / 2))
    assert m.idf["c"] == pytest.approx(math.log10(3))
    assert m.idf["d"] == pytest.approx(math.log10(3))


def test_smooth_idf_formula():
    m = full_model(smooth_idf=True)
    assert m.idf["a"] == pytest.approx(math.log10(4 / 3) + 1)


def test_ltc_weights_before_normalisation():
    m = full_model(normalize=False)
    idf_a, idf_b, idf_d = math.log10(1.5), math.log10(1.5), math.log10(3)
    X = m.matrix
    col = m.term_to_col
    assert X[0, col["a"]] == pytest.approx((1 + math.log10(2)) * idf_a, rel=1e-6)
    assert X[0, col["b"]] == pytest.approx(1 * idf_b, rel=1e-6)
    assert X[2, col["d"]] == pytest.approx((1 + math.log10(3)) * idf_d, rel=1e-6)
    assert X[0, col["c"]] == 0.0


def test_raw_tf_when_not_sublinear():
    m = full_model(normalize=False, sublinear_tf=False)
    assert m.matrix[2, m.term_to_col["d"]] == pytest.approx(3 * math.log10(3), rel=1e-6)


def test_l2_normalised_rows_hand_value():
    m = full_model()
    wa, wb = (1 + math.log10(2)) * math.log10(1.5), math.log10(1.5)
    norm = math.sqrt(wa ** 2 + wb ** 2)
    assert m.matrix[0, m.term_to_col["a"]] == pytest.approx(wa / norm, rel=1e-6)
    assert m.matrix[0, m.term_to_col["b"]] == pytest.approx(wb / norm, rel=1e-6)


def test_rows_have_unit_length_on_real_data(baseline_tfidf):
    norms = np.linalg.norm(baseline_tfidf.matrix, axis=1)
    nonzero = norms > 0
    assert nonzero.mean() > 0.99
    assert np.allclose(norms[nonzero], 1.0, atol=1e-5)


def test_tf_weight_zero():
    assert TfidfModel()._tf_weight(0) == 0.0


# ------------------------------------------------------------------ pruning
def test_min_df_pruning():
    m = TfidfModel(min_df=2, max_df_ratio=1.0, max_features=None).fit(DOCS)
    assert m.vocab == ["a", "b"]
    assert m.pruned_stats["removed_rare(<min_df)"] == 2
    assert m.pruned_stats["raw_terms"] == 4 and m.pruned_stats["kept_terms"] == 2


def test_max_df_pruning():
    docs = [["x", "a"], ["x", "b"], ["x", "c"], ["x", "a"]]
    m = TfidfModel(min_df=1, max_df_ratio=0.9, max_features=None).fit(docs)
    assert "x" not in m.vocab          # df 4 > 0.9*4
    assert m.pruned_stats["removed_common(>max_df)"] == 1


def test_max_features_keeps_most_frequent():
    # collection frequencies: d=3, a=3, b=2, c=1 -> tie a/d broken alphabetically
    m = TfidfModel(min_df=1, max_df_ratio=1.0, max_features=2).fit(DOCS)
    assert m.vocab == ["a", "d"]
    m3 = TfidfModel(min_df=1, max_df_ratio=1.0, max_features=3).fit(DOCS)
    assert m3.vocab == ["a", "b", "d"]


def test_real_data_pruning(baseline_tfidf):
    s = baseline_tfidf.pruned_stats
    assert s["raw_terms"] == 21238 and s["kept_terms"] == 5000
    assert baseline_tfidf.matrix.shape == (2225, 5000)


# ------------------------------------------------------------------ transform
def test_transform_training_doc_equals_matrix_row():
    m = full_model()
    for i, d in enumerate(DOCS):
        assert np.allclose(m.transform([d])[0], m.matrix[i], atol=1e-6)


def test_transform_training_doc_real_data(baseline_tfidf, baseline_tokens):
    for i in (0, 500, 2224):
        assert np.allclose(baseline_tfidf.transform([baseline_tokens[i]])[0], baseline_tfidf.matrix[i], atol=1e-5)


def test_transform_ignores_unknown_terms():
    m = full_model()
    v = m.transform([["a", "a", "b", "unknown", "zzz"]])[0]
    assert np.allclose(v, m.matrix[0], atol=1e-6)
    assert not m.transform([["unknown"]])[0].any()


def test_top_terms_and_idf_table():
    m = full_model()
    assert m.top_terms(2, 5)[0][0] == "d"
    assert all(w > 0 for _, w in m.top_terms(0, 10))
    tbl = m.idf_table()
    assert tbl[0][2] >= tbl[-1][2]
    assert {t: df for t, df, _ in tbl} == {"a": 2, "b": 2, "c": 1, "d": 1}


# ------------------------------------------------------------------ cosine
def test_cosine_identical_and_disjoint():
    a = np.array([1.0, 2.0, 0.0])
    assert cosine_similarity(a, a)[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, 3 * a)[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, np.array([0.0, 0.0, 5.0]))[0, 0] == 0.0
    assert cosine_similarity(a, np.zeros(3))[0, 0] == 0.0
    assert cosine_pair_scalar({"x": 1.0}, {"y": 1.0}) == 0.0
    assert cosine_pair_scalar({"x": 1.0, "y": 2.0}, {"x": 1.0, "y": 2.0}) == pytest.approx(1.0)
    assert cosine_pair_scalar({}, {"x": 1.0}) == 0.0


def test_cosine_matrix_matches_pure_python():
    m = full_model(normalize=False)
    S = cosine_similarity(m.matrix, m.matrix)
    dicts = [{t: float(m.matrix[i, c]) for t, c in m.term_to_col.items() if m.matrix[i, c]} for i in range(N)]
    for i in range(N):
        for j in range(N):
            assert S[i, j] == pytest.approx(cosine_pair_scalar(dicts[i], dicts[j]), abs=1e-6)


def test_cosine_random_vectors_match_pure_python():
    rng = np.random.default_rng(0)
    A = rng.random((4, 6)) * (rng.random((4, 6)) > 0.4)
    S = cosine_similarity(A, A)
    for i in range(4):
        for j in range(4):
            u = {k: float(A[i, k]) for k in range(6) if A[i, k]}
            v = {k: float(A[j, k]) for k in range(6) if A[j, k]}
            assert S[i, j] == pytest.approx(cosine_pair_scalar(u, v), abs=1e-6)


def test_cosine_equals_dot_for_normalised_rows():
    m = full_model()
    assert np.allclose(cosine_similarity(m.matrix, m.matrix), m.matrix @ m.matrix.T, atol=1e-6)
