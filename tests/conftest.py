"""Shared pytest fixtures.

Expensive objects (the 2,225-document BBC corpus, its baseline preprocessing,
TF-IDF model and the full `run_experiment` results) are built ONCE per test
session so the whole suite stays fast.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from textmining.data_loader import load_default  # noqa: E402
from textmining.pipeline import run_experiment  # noqa: E402
from textmining.preprocessing import PreprocessConfig, preprocess_corpus  # noqa: E402
from textmining.tfidf import TfidfModel  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def dataset():
    """(documents_df, metadata_df) of the default BBC corpus."""
    return load_default()


@pytest.fixture(scope="session")
def texts(dataset):
    return dataset[0]["text"].fillna("").astype(str).tolist()


@pytest.fixture(scope="session")
def labels(dataset):
    return dataset[1]["category"].astype(str).tolist()


@pytest.fixture(scope="session")
def baseline_tokens(texts):
    return preprocess_corpus(texts, PreprocessConfig())


@pytest.fixture(scope="session")
def baseline_tfidf(baseline_tokens):
    return TfidfModel(min_df=2, max_df_ratio=0.9, max_features=5000, sublinear_tf=True).fit(baseline_tokens)


@pytest.fixture(scope="session")
def baseline_run(dataset, baseline_tokens):
    docs, meta = dataset
    return run_experiment(docs, meta, PreprocessConfig(), tokens=baseline_tokens)


@pytest.fixture(scope="session")
def dedup_run(dataset, baseline_tokens):
    docs, meta = dataset
    return run_experiment(docs, meta, PreprocessConfig(), tokens=baseline_tokens, dedup=True)
