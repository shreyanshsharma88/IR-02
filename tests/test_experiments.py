"""Sanity checks on the OUTPUTS of experiments.py (report/results.json + figures).

experiments.py itself is NOT executed here (slow, and it may be re-run concurrently);
tests skip gracefully when the outputs are missing.
"""
from __future__ import annotations

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "report", "results.json")
FIG_DIR = os.path.join(ROOT, "report", "figures")


@pytest.fixture(scope="module")
def results():
    if not os.path.exists(RESULTS):
        pytest.skip("report/results.json not generated yet (run experiments.py)")
    try:
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        pytest.skip("results.json is being rewritten right now")


def test_top_level_keys(results):
    assert {"seed", "categories", "dataset", "duplicates", "preprocessing", "tfidf", "naive_bayes", "kmeans",
            "stability", "comparison", "recommendation", "ablation", "timings_s", "figures"} <= set(results)


def test_dataset_section(results):
    d = results["dataset"]
    assert d["n_docs"] == 2225 and d["n_categories"] == 5
    assert sum(d["docs_per_category"].values()) == 2225
    assert results["categories"] == ["business", "entertainment", "politics", "sport", "tech"]


def test_duplicates_section(results):
    assert results["duplicates"]["exact_groups"] == 137


def test_tfidf_section(results):
    t = results["tfidf"]
    assert t["matrix_shape"] == [2225, 5000]
    assert t["pruning"]["kept_terms"] == 5000


def test_comparison_values_sane(results):
    c = results["comparison"]
    for k, v in c.items():
        assert 0.0 <= v <= 1.0, k
    assert c["NB accuracy"] > 0.93
    assert c["KM purity"] > 0.9 and c["KM Rand index"] > 0.9


def test_ablation_rows(results):
    ab = results["ablation"]
    assert len(ab) >= 7
    for row in ab:
        assert 0 < row["nb_accuracy"] <= 1 and 0 < row["km_purity"] <= 1 and 0 < row["km_rand_index"] <= 1
        if row["dedup"]:
            assert row["test_leakage_docs"] == 0 and row["docs_used"] < 2225
    base = next(r for r in ab if r["config"] == "lower|stop|stem|ng1-1" and not r["dedup"])
    assert base["test_leakage_docs"] == 38 and base["vocab_raw"] == 21238


def test_recommendation_queries(results):
    qs = results["recommendation"]["queries"]
    assert qs
    for q in qs:
        scores = [r["score"] for r in q["top5"]]
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= s <= 1.0001 for s in scores)


def test_figures_exist(results):
    figs = results["figures"]
    assert len(figs) >= 10
    missing = [f for f in figs if not os.path.isfile(os.path.join(FIG_DIR, os.path.basename(f)))]
    assert not missing, missing
    for f in figs:
        with open(os.path.join(FIG_DIR, os.path.basename(f)), "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("name", ["kmeans_membership.csv", "nb_test_predictions.csv"])
def test_csv_outputs(name):
    p = os.path.join(ROOT, "report", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not generated")
    import pandas as pd
    df = pd.read_csv(p)
    assert len(df) > 0
