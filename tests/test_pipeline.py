"""Integration tests on the full BBC corpus: run_experiment end-to-end + dataset regeneration."""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys

import pytest

from textmining.pipeline import leakage, run_experiment
from textmining.preprocessing import PreprocessConfig

EXPECTED_KEYS = {"config", "dedup", "docs_used", "dropped_duplicates", "vocab_raw", "vocab_tfidf", "avg_tokens",
                 "nb_accuracy", "nb_macro_f1", "nb_macro_precision", "nb_macro_recall", "test_leakage_docs",
                 "km_purity", "km_rand_index", "km_sse", "km_iterations", "precision_at_5", "timings"}


def test_dataset_size(dataset):
    docs, meta = dataset
    assert len(docs) >= 2225
    total = docs["text"].str.len().sum() + meta["title"].str.len().sum()
    assert 4.5e6 < total < 6e6                     # ~5 MB of text
    assert meta["category"].nunique() == 5


def test_baseline_run(baseline_run):
    r = baseline_run
    assert set(r) == EXPECTED_KEYS
    assert r["config"] == "lower|stop|stem|ng1-1"
    assert r["docs_used"] == 2225 and r["dropped_duplicates"] == 0
    assert r["vocab_raw"] == 21238 and r["vocab_tfidf"] == 5000
    assert r["nb_accuracy"] > 0.93
    assert r["nb_macro_f1"] > 0.93
    assert r["km_purity"] > 0.9
    assert r["km_rand_index"] > 0.9
    assert r["test_leakage_docs"] == 38
    assert r["precision_at_5"] > 0.85
    assert set(r["timings"]) == {"preprocess_s", "tfidf_s", "naive_bayes_s", "kmeans_s", "recommend_eval_s"}


def test_baseline_matches_report_numbers(baseline_run):
    assert round(baseline_run["nb_accuracy"], 4) == 0.9617
    assert round(baseline_run["km_purity"], 4) == 0.9667
    assert round(baseline_run["km_rand_index"], 4) == 0.9749


def test_dedup_run(dedup_run):
    r = dedup_run
    assert r["dedup"] is True
    assert r["docs_used"] == 2063 and r["dropped_duplicates"] == 162
    assert r["test_leakage_docs"] == 0
    assert r["nb_accuracy"] > 0.95


def test_run_is_deterministic(dataset, baseline_tokens, baseline_run):
    docs, meta = dataset
    again = run_experiment(docs, meta, PreprocessConfig(), tokens=baseline_tokens)
    for k in EXPECTED_KEYS - {"timings"}:
        assert again[k] == baseline_run[k], k


def test_run_preprocesses_when_no_tokens_given(dataset):
    docs, meta = dataset
    sub_d, sub_m = docs.iloc[::10].reset_index(drop=True), meta.iloc[::10].reset_index(drop=True)
    r = run_experiment(sub_d, sub_m, PreprocessConfig(normalizer="none"), n_init=1)
    assert r["docs_used"] == len(sub_d) and r["config"] == "lower|stop|none|ng1-1"
    assert r["nb_accuracy"] > 0.8


def test_leakage_function():
    texts = ["Same text", "same   TEXT", "other", "unique"]
    assert leakage([0, 2], [1, 3], texts) == 1
    assert leakage([2], [3], texts) == 0


# ------------------------------------------------------------------ data/prepare_dataset.py
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_download", "bbc")


@pytest.mark.skipif(not os.path.isdir(RAW), reason="raw BBC folder not present")
def test_prepare_dataset_regenerates_identical_csvs(project_root, tmp_path):
    """The script writes next to itself, so run a copy inside tmp_path against the raw folder."""
    shutil.copy(os.path.join(project_root, "data", "prepare_dataset.py"), tmp_path / "prepare_dataset.py")
    out = subprocess.run([sys.executable, str(tmp_path / "prepare_dataset.py"), RAW],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "wrote 2225 documents" in out.stdout
    for name in ("documents.csv", "metadata.csv"):
        with open(os.path.join(project_root, "data", name), "rb") as a, open(tmp_path / name, "rb") as b:
            assert a.read() == b.read(), f"{name} differs from a fresh regeneration"


@pytest.mark.skipif(not os.path.isdir(RAW), reason="raw BBC folder not present")
def test_csv_content_matches_raw_files(project_root):
    with open(os.path.join(project_root, "data", "documents.csv"), encoding="utf-8", newline="") as f:
        docs = {r["doc_id"]: r["text"] for r in csv.DictReader(f)}
    with open(os.path.join(project_root, "data", "metadata.csv"), encoding="utf-8", newline="") as f:
        meta = list(csv.DictReader(f))
    for m in meta[::150]:
        with open(os.path.join(RAW, m["filename"]), encoding="utf-8") as f:
            raw = f.read()
        lines = raw.splitlines()
        assert m["title"] == lines[0].strip()
        assert docs[m["doc_id"]] == "\n".join(lines[1:]).strip()
        assert m["category"] == m["filename"].split("/")[0]
        assert int(m["n_chars"]) == len(raw)
