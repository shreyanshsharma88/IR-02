"""K-Means from scratch: recovery of synthetic blobs, convergence log, determinism, robustness."""
from __future__ import annotations

import numpy as np
import pytest

from textmining.kmeans import KMeans, _sq_dists, elbow
from textmining.metrics import purity, rand_index


def blobs(n_per=30, k=4, dim=5, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.eye(k, dim) * 10
    X = np.vstack([c + rng.normal(0, spread, (n_per, dim)) for c in centres])
    y = np.repeat(np.arange(k), n_per)
    return X.astype(np.float32), y


def test_sq_dists():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    C = np.array([[0.0, 0.0], [3.0, 0.0]])
    assert np.allclose(_sq_dists(X, C), [[0, 9], [25, 16]])


@pytest.mark.parametrize("init", ["kmeans++", "random"])
def test_recovers_separated_blobs(init):
    X, y = blobs()
    km = KMeans(k=4, init=init, n_init=5, seed=3).fit(X)
    assert purity(km.labels_.tolist(), y.tolist()) == 1.0
    assert rand_index(km.labels_.tolist(), y.tolist()) == 1.0
    assert sorted(np.bincount(km.labels_)) == [30, 30, 30, 30]
    assert km.centroids.shape == (4, 5)
    assert np.array_equal(km.predict(X), km.labels_)


def test_sse_non_increasing_and_log_fields():
    rng = np.random.default_rng(1)
    X = rng.random((200, 8)).astype(np.float32)
    km = KMeans(k=6, n_init=3, seed=11, max_iter=100, tol=0).fit(X)
    sses = [h["SSE"] for h in km.history]
    assert all(b <= a + 1e-3 for a, b in zip(sses, sses[1:]))
    assert km.history[0]["iteration"] == 1 and len(km.history) == km.n_iter_
    for h in km.history:
        assert set(h) >= {"iteration", "docs_changed", "max_centroid_shift", "SSE", "cluster_sizes"}
        assert sum(h["cluster_sizes"]) == 200
    assert km.sse_ == pytest.approx(sses[-1], rel=1e-3)


def test_stop_reason_set():
    X, _ = blobs()
    km = KMeans(k=4, n_init=1, seed=0).fit(X)
    assert km.stop_reason.startswith("converged")
    km1 = KMeans(k=4, n_init=1, seed=0, max_iter=1, tol=0).fit(np.random.default_rng(0).random((100, 3)))
    assert "max_iter" in km1.stop_reason and km1.n_iter_ == 1


def test_stop_on_tolerance():
    X = np.random.default_rng(2).random((300, 4)).astype(np.float32)
    km = KMeans(k=5, n_init=1, seed=2, tol=0.5).fit(X)
    assert "tol" in km.stop_reason or "no document changed" in km.stop_reason


def test_deterministic_for_seed():
    X = np.random.default_rng(5).random((150, 6)).astype(np.float32)
    a = KMeans(k=4, seed=9, n_init=3).fit(X)
    b = KMeans(k=4, seed=9, n_init=3).fit(X)
    assert np.array_equal(a.labels_, b.labels_)
    assert a.sse_ == b.sse_ and a.init_indices == b.init_indices and a.run_summary == b.run_summary


@pytest.mark.parametrize("init", ["kmeans++", "random"])
def test_init_indices(init):
    X = np.random.default_rng(4).random((50, 3)).astype(np.float32)
    km = KMeans(k=7, init=init, n_init=1, seed=0).fit(X)
    assert len(km.init_indices) == 7
    assert all(0 <= i < 50 for i in km.init_indices)
    if init == "random":
        assert len(set(km.init_indices)) == 7


def test_kmeanspp_picks_distinct_points_on_blobs():
    X, y = blobs(k=4)
    km = KMeans(k=4, init="kmeans++", n_init=1, seed=0).fit(X)
    assert len({int(y[i]) for i in km.init_indices}) == 4    # D^2 sampling spreads seeds across blobs


def test_empty_cluster_robustness():
    # many identical points: random init picks identical centroids -> empty clusters must be re-seeded
    X = np.vstack([np.zeros((20, 2)), np.ones((2, 2)) * 5]).astype(np.float32)
    for seed in range(10):
        km = KMeans(k=3, init="random", n_init=1, seed=seed, max_iter=20).fit(X)
        assert not np.isnan(km.centroids).any()
        assert km.labels_.min() >= 0 and km.labels_.max() < 3
        assert np.isfinite(km.sse_)


def test_n_init_keeps_lowest_sse():
    X = np.random.default_rng(8).random((120, 5)).astype(np.float32)
    km = KMeans(k=6, n_init=6, init="random", seed=1).fit(X)
    assert len(km.run_summary) == 6
    assert [r["run"] for r in km.run_summary] == list(range(1, 7))
    assert round(km.sse_, 4) == min(r["SSE"] for r in km.run_summary)


def test_top_terms():
    X, _ = blobs(k=3, dim=3)
    km = KMeans(k=3, n_init=1, seed=0).fit(X)
    tt = km.top_terms(["x", "y", "z"], 1)
    assert sorted(v[0] for v in tt.values()) == ["x", "y", "z"]


def test_elbow_decreasing():
    X, _ = blobs()
    res = elbow(X, [1, 2, 4], seed=0)
    assert [k for k, _ in res] == [1, 2, 4]
    sse = [s for _, s in res]
    assert sse[0] > sse[1] > sse[2]


def test_real_data_kmeans(baseline_run):
    assert baseline_run["km_iterations"] == 31
    assert baseline_run["km_sse"] == pytest.approx(2072.77, abs=0.5)
