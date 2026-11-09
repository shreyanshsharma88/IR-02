"""Session state + cached pipeline shared by every page.

The pipeline is lazy: each page asks the `Pipeline` context for what it needs
(tokens, TF-IDF, duplicates, Naive Bayes, K-Means, ...).  Every expensive step
is cached with `st.cache_resource`, keyed on (dataset key, JSON of the
relevant configuration).  Large inputs are passed as underscore-prefixed
arguments so Streamlit does not hash them.  On first load the defaults are
used, so every page works immediately.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from dataclasses import asdict
from functools import cached_property

import numpy as np
import pandas as pd
import streamlit as st

from textmining import metrics as M
from textmining.data_loader import load_default, load_uploaded
from textmining.duplicates import duplicate_report, exact_duplicate_groups
from textmining.kmeans import KMeans
from textmining.naive_bayes import MultinomialNaiveBayes, stratified_split
from textmining.pca import pca_2d
from textmining.pipeline import leakage
from textmining.preprocessing import PreprocessConfig, corpus_stats, preprocess_corpus, raw_tokens
from textmining.recommender import ContentRecommender, precision_at_k
from textmining.tfidf import TfidfModel

DEFAULT_TFIDF = dict(min_df=2, max_df=0.9, max_features=5000, sublinear=True)
DEFAULT_NB = dict(test_size=0.2, alpha=1.0, seed=42)
DEFAULT_KM = dict(k=5, init="kmeans++", max_iter=100, tol=1e-4, n_init=5, seed=42)
DEFAULT_DATASET_KEY = "bbc-default"


def _j(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


# ===================================================================== dataset
def _ensure_meta(docs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    if "category" not in meta.columns:
        meta["category"] = "unlabelled"
    meta["category"] = meta["category"].fillna("unlabelled").astype(str)
    if "title" not in meta.columns:
        meta["title"] = docs["text"].astype(str).str.slice(0, 70)
    meta["title"] = meta["title"].fillna("").astype(str)
    for col, default in (("source", "—"), ("period", "—")):
        if col not in meta.columns:
            meta[col] = default
        meta[col] = meta[col].fillna(default).astype(str)
    if "n_chars" not in meta.columns:
        meta["n_chars"] = docs["text"].astype(str).str.len()
    if "n_words" not in meta.columns:
        meta["n_words"] = docs["text"].astype(str).str.split().str.len()
    return meta.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading BBC News corpus ...")
def _load_default_cached():
    docs, meta = load_default()
    docs["text"] = docs["text"].fillna("").astype(str)
    return docs, _ensure_meta(docs, meta)


def load_upload(name: str, data: bytes) -> dict:
    docs, meta = load_uploaded(name, data)
    docs = docs.reset_index(drop=True)
    docs["text"] = docs["text"].fillna("").astype(str)
    meta = _ensure_meta(docs, meta)
    key = "upload:" + hashlib.md5(data).hexdigest()
    return {"key": key, "name": name, "docs": docs, "meta": meta}


def default_dataset() -> dict:
    docs, meta = _load_default_cached()
    return {"key": DEFAULT_DATASET_KEY, "name": "BBC News (default, 2,225 articles)", "docs": docs, "meta": meta}


def init_state():
    ss = st.session_state
    if "ds" not in ss:
        ss.ds = default_dataset()
    ss.setdefault("cfg", PreprocessConfig())
    ss.setdefault("tfidf_params", dict(DEFAULT_TFIDF))
    ss.setdefault("dedup", False)
    ss.setdefault("dup_threshold", 0.9)
    ss.setdefault("nb_params", dict(DEFAULT_NB))
    ss.setdefault("km_params", dict(DEFAULT_KM))


def set_dataset(ds: dict):
    ss = st.session_state
    ss.ds = ds
    ss.dedup = False
    for k in ("ablation", "seed_stability", "elbow"):
        ss.pop(k, None)


def persist(src: str, dst: str):
    """on_change callback: copy a widget value into a persistent session key."""
    st.session_state[dst] = st.session_state[src]


# ===================================================================== cached steps
@st.cache_resource(show_spinner="Tokenising raw text ...", max_entries=3)
def _raw_tokens(ds_key, _texts):
    return [raw_tokens(t) for t in _texts]


@st.cache_resource(show_spinner=False, max_entries=6)
def _stats(ds_key, which, _token_lists):
    return corpus_stats(_token_lists)


@st.cache_resource(show_spinner="Preprocessing corpus ...", max_entries=6)
def _tokens(ds_key, cfg_json, _texts):
    t = time.perf_counter()
    toks = preprocess_corpus(_texts, PreprocessConfig(**json.loads(cfg_json)))
    return toks, time.perf_counter() - t


@st.cache_resource(show_spinner="Building inverted index + TF-IDF ...", max_entries=6)
def _tfidf(ds_key, cfg_json, tfidf_json, subset_key, _toks):
    p = json.loads(tfidf_json)
    t = time.perf_counter()
    model = TfidfModel(min_df=p["min_df"], max_df_ratio=p["max_df"], max_features=p["max_features"] or None,
                       sublinear_tf=p["sublinear"]).fit(_toks)
    return model, time.perf_counter() - t


@st.cache_resource(show_spinner="Detecting duplicates ...", max_entries=6)
def _dups(ds_key, cfg_json, tfidf_json, threshold, _texts, _matrix):
    t = time.perf_counter()
    rep = duplicate_report(_texts, _matrix, threshold)
    return rep, time.perf_counter() - t


@st.cache_resource(show_spinner=False, max_entries=4)
def _exact_groups(ds_key, _texts):
    return exact_duplicate_groups(_texts)


@st.cache_resource(show_spinner="Training Multinomial Naive Bayes ...", max_entries=6)
def _nb(ds_key, cfg_json, subset_key, nb_json, _toks, _labels, _texts):
    p = json.loads(nb_json)
    tr, te = stratified_split(_labels, p["test_size"], p["seed"])
    t = time.perf_counter()
    model = MultinomialNaiveBayes(p["alpha"]).fit([_toks[i] for i in tr], [_labels[i] for i in tr])
    pred = model.predict([_toks[i] for i in te])
    elapsed = time.perf_counter() - t
    y_true = [_labels[i] for i in te]
    return {"model": model, "train_idx": tr, "test_idx": te, "y_true": y_true, "y_pred": pred,
            "report": M.classification_report(y_true, pred), "leakage": leakage(tr, te, _texts),
            "seconds": elapsed}


@st.cache_resource(show_spinner="Running K-Means ...", max_entries=6)
def _km(ds_key, cfg_json, tfidf_json, subset_key, km_json, _X):
    p = json.loads(km_json)
    t = time.perf_counter()
    km = KMeans(k=p["k"], init=p["init"], max_iter=p["max_iter"], tol=p["tol"], n_init=p["n_init"],
                seed=p["seed"]).fit(_X)
    return km, time.perf_counter() - t


@st.cache_resource(show_spinner="Projecting to 2-D (PCA, visualisation only) ...", max_entries=4)
def _pca(ds_key, cfg_json, tfidf_json, subset_key, _X):
    t = time.perf_counter()
    return pca_2d(_X), time.perf_counter() - t


@st.cache_resource(show_spinner="Evaluating Precision@K over a sample ...", max_entries=12)
def _p_at_k(ds_key, cfg_json, tfidf_json, subset_key, k, dup_threshold, _rec, _labels):
    n = len(_labels)
    sample = list(range(0, n, max(1, n // 300)))
    t = time.perf_counter()
    val = precision_at_k(_rec, _labels, k, sample, dup_threshold)
    return val, len(sample), time.perf_counter() - t


# ===================================================================== pipeline context
class Pipeline:
    """Lazy view of the whole pipeline for the current session configuration."""

    def __init__(self):
        ss = st.session_state
        self.ds = ss.ds
        self.key = ss.ds["key"]
        self.docs: pd.DataFrame = ss.ds["docs"]
        self.meta: pd.DataFrame = ss.ds["meta"]
        self.cfg: PreprocessConfig = ss.cfg
        self.cfg_json = _j(asdict(self.cfg))
        self.tfidf_params = ss.tfidf_params
        self.tfidf_json = _j(self.tfidf_params)
        self.dedup = bool(ss.dedup)
        self.dup_threshold = float(ss.dup_threshold)
        self.nb_params = ss.nb_params
        self.km_params = ss.km_params
        self.subset_key = f"dedup@{self.dup_threshold:.3f}" if self.dedup else "all"

    # ------------------------------------------------------------ raw
    @cached_property
    def texts_all(self) -> list[str]:
        return self.docs["text"].tolist()

    @cached_property
    def labels_all(self) -> list[str]:
        return self.meta["category"].astype(str).tolist()

    @cached_property
    def raw_toks(self):
        return _raw_tokens(self.key, self.texts_all)

    @cached_property
    def raw_stats(self):
        return _stats(self.key, "raw", self.raw_toks)

    @cached_property
    def exact_groups(self):
        return _exact_groups(self.key, self.texts_all)

    # ------------------------------------------------------------ preprocessing (all docs)
    @cached_property
    def _tok_res(self):
        return _tokens(self.key, self.cfg_json, self.texts_all)

    @property
    def tokens_all(self):
        return self._tok_res[0]

    @cached_property
    def proc_stats_all(self):
        return _stats(self.key, "proc:" + self.cfg_json, self.tokens_all)

    @cached_property
    def _tfidf_full_res(self):
        return _tfidf(self.key, self.cfg_json, self.tfidf_json, "all", self.tokens_all)

    @property
    def tfidf_full(self) -> TfidfModel:
        return self._tfidf_full_res[0]

    @cached_property
    def _dup_res(self):
        return _dups(self.key, self.cfg_json, self.tfidf_json, round(self.dup_threshold, 3),
                     self.texts_all, self.tfidf_full.matrix)

    @property
    def dups(self) -> dict:
        return self._dup_res[0]

    # ------------------------------------------------------------ working set
    @cached_property
    def work_idx(self) -> list[int]:
        if not self.dedup:
            return list(range(len(self.texts_all)))
        drop = set(self.dups["drop_indices"])
        return [i for i in range(len(self.texts_all)) if i not in drop]

    @cached_property
    def tokens(self):
        if not self.dedup:
            return self.tokens_all
        return [self.tokens_all[i] for i in self.work_idx]

    @cached_property
    def texts(self):
        return [self.texts_all[i] for i in self.work_idx]

    @cached_property
    def meta_w(self) -> pd.DataFrame:
        return self.meta.iloc[self.work_idx].reset_index(drop=True)

    @cached_property
    def labels(self) -> list[str]:
        return self.meta_w["category"].astype(str).tolist()

    @cached_property
    def _tfidf_res(self):
        if not self.dedup:
            return self._tfidf_full_res
        return _tfidf(self.key, self.cfg_json, self.tfidf_json, self.subset_key, self.tokens)

    @property
    def tfidf(self) -> TfidfModel:
        return self._tfidf_res[0]

    @property
    def X(self) -> np.ndarray:
        return self.tfidf.matrix

    @cached_property
    def n_classes(self) -> int:
        return len(set(self.labels))

    # ------------------------------------------------------------ models
    @cached_property
    def nb(self) -> dict:
        return _nb(self.key, self.cfg_json, self.subset_key, _j(self.nb_params), self.tokens, self.labels, self.texts)

    @cached_property
    def _km_res(self):
        return _km(self.key, self.cfg_json, self.tfidf_json, self.subset_key, _j(self.km_params), self.X)

    @property
    def km(self) -> KMeans:
        return self._km_res[0]

    @cached_property
    def km_eval(self) -> dict:
        cl = self.km.labels_.tolist()
        counts = M.rand_index_counts(cl, self.labels)
        return {"purity": M.purity(cl, self.labels), "rand_index": counts["rand_index"], "counts": counts,
                "majority": M.majority_label_map(cl, self.labels)}

    @cached_property
    def _pca_res(self):
        return _pca(self.key, self.cfg_json, self.tfidf_json, self.subset_key, self.X)

    @property
    def pca(self) -> np.ndarray:
        return self._pca_res[0]

    @cached_property
    def recommender(self) -> ContentRecommender:
        return ContentRecommender(self.tfidf, self.meta_w)

    def p_at_k(self, k: int, dup_threshold: float | None):
        thr = None if dup_threshold is None else round(float(dup_threshold), 3)
        return _p_at_k(self.key, self.cfg_json, self.tfidf_json, self.subset_key, int(k), thr,
                       self.recommender, self.labels)

    # ------------------------------------------------------------ misc
    def timings(self) -> dict:
        """Elapsed seconds of each stage (as measured when it was computed)."""
        out = {"Preprocessing": self._tok_res[1], "Inverted index + TF-IDF": self._tfidf_res[1]}
        if self.dedup:
            out["TF-IDF (full corpus, for dedup)"] = self._tfidf_full_res[1]
        out["Duplicate detection"] = self._dup_res[1]
        out["Naive Bayes (train + predict)"] = self.nb["seconds"]
        out["K-Means (all restarts)"] = self._km_res[1]
        out["PCA (2-D, visual only)"] = self._pca_res[1]
        return out

    def doc_options(self) -> list[str]:
        return [f"{d} — {t}" for d, t in zip(self.meta_w["doc_id"].astype(str), self.meta_w["title"])]

    def sample_balanced(self, per_class: int = 4, seed: int = 7) -> list[int]:
        rng = random.Random(seed)
        by = {}
        for i, y in enumerate(self.labels):
            by.setdefault(y, []).append(i)
        out = []
        for y in sorted(by):
            out.extend(sorted(rng.sample(by[y], min(per_class, len(by[y])))))
        return out

    def category_counts(self) -> pd.Series:
        return pd.Series(Counter(self.labels_all)).sort_index()


def get_pipeline() -> Pipeline:
    init_state()
    return Pipeline()
