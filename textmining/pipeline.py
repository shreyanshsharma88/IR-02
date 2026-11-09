"""End-to-end experiment runner shared by the Streamlit app and experiments.py."""
from __future__ import annotations

import time

import numpy as np

from . import metrics as M
from .duplicates import duplicate_report
from .kmeans import KMeans
from .naive_bayes import MultinomialNaiveBayes, stratified_split
from .preprocessing import PreprocessConfig, corpus_stats, preprocess_corpus
from .recommender import ContentRecommender, precision_at_k
from .tfidf import TfidfModel


def leakage(train_idx, test_idx, texts):
    """# test docs whose (normalised) text appears verbatim in the training set."""
    norm = lambda t: " ".join(str(t).lower().split())
    train_set = {norm(texts[i]) for i in train_idx}
    return sum(norm(texts[i]) in train_set for i in test_idx)


def run_experiment(docs, meta, cfg: PreprocessConfig, *, min_df=2, max_df=0.9, max_features=5000,
                   sublinear=True, dedup=False, dup_threshold=0.9, k=5, seed=42, test_size=0.2,
                   alpha=1.0, init="kmeans++", n_init=5, p_at_k=5, tokens=None) -> dict:
    timings = {}
    texts_all = docs["text"].tolist()
    labels_all = meta["category"].astype(str).tolist()

    t = time.perf_counter()
    toks_all = tokens if tokens is not None else preprocess_corpus(texts_all, cfg)
    timings["preprocess_s"] = time.perf_counter() - t

    keep = list(range(len(texts_all)))
    n_dropped = 0
    if dedup:
        tmp = TfidfModel(min_df=min_df, max_df_ratio=max_df, max_features=max_features, sublinear_tf=sublinear).fit(toks_all)
        rep = duplicate_report(texts_all, tmp.matrix, dup_threshold)
        drop = set(rep["drop_indices"])
        keep = [i for i in keep if i not in drop]
        n_dropped = len(drop)
    toks = [toks_all[i] for i in keep]
    labels = [labels_all[i] for i in keep]
    texts = [texts_all[i] for i in keep]

    t = time.perf_counter()
    tf = TfidfModel(min_df=min_df, max_df_ratio=max_df, max_features=max_features, sublinear_tf=sublinear).fit(toks)
    timings["tfidf_s"] = time.perf_counter() - t

    tr, te = stratified_split(labels, test_size, seed)
    t = time.perf_counter()
    nb = MultinomialNaiveBayes(alpha).fit([toks[i] for i in tr], [labels[i] for i in tr])
    pred = nb.predict([toks[i] for i in te])
    timings["naive_bayes_s"] = time.perf_counter() - t
    rep = M.classification_report([labels[i] for i in te], pred)

    t = time.perf_counter()
    km = KMeans(k=k, init=init, n_init=n_init, seed=seed).fit(tf.matrix)
    timings["kmeans_s"] = time.perf_counter() - t
    cl = km.labels_.tolist()

    rec = ContentRecommender(tf, meta.iloc[keep])
    sample = list(range(0, len(labels), max(1, len(labels) // 300)))
    t = time.perf_counter()
    p5 = precision_at_k(rec, labels, p_at_k, sample)
    timings["recommend_eval_s"] = time.perf_counter() - t

    st = corpus_stats(toks)
    return {
        "config": cfg.label(), "dedup": dedup, "docs_used": len(keep), "dropped_duplicates": n_dropped,
        "vocab_raw": st["vocabulary_size"], "vocab_tfidf": len(tf.vocab), "avg_tokens": st["avg_tokens_per_doc"],
        "nb_accuracy": rep["accuracy"], "nb_macro_f1": rep["macro"]["f1"],
        "nb_macro_precision": rep["macro"]["precision"], "nb_macro_recall": rep["macro"]["recall"],
        "test_leakage_docs": leakage(tr, te, texts),
        "km_purity": M.purity(cl, labels), "km_rand_index": M.rand_index(cl, labels), "km_sse": km.sse_,
        "km_iterations": km.n_iter_, f"precision_at_{p_at_k}": p5,
        "timings": timings,
    }
