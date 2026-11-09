"""Reproducible experiment runner for the report.

    python experiments.py            # ~ a few minutes on a laptop

Writes  report/results.json  (every number quoted in the report)
        report/figures/*.png (every figure embedded in the report)
        report/nb_test_predictions.csv, report/kmeans_membership.csv (full tables)

All algorithms (TF-IDF, cosine, Naive Bayes, K-Means, Purity, Rand Index) come from the
hand-written `textmining` package. No scikit-learn is used anywhere. Seed = 42.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from textmining import metrics as M  # noqa: E402
from textmining.data_loader import load_default  # noqa: E402
from textmining.duplicates import duplicate_report, jaccard, shingles  # noqa: E402
from textmining.kmeans import KMeans, elbow  # noqa: E402
from textmining.naive_bayes import MultinomialNaiveBayes, stratified_split  # noqa: E402
from textmining.pca import pca_2d  # noqa: E402
from textmining.pipeline import leakage, run_experiment  # noqa: E402
from textmining.preprocessing import (PreprocessConfig, corpus_stats, preprocess,  # noqa: E402
                                      preprocess_corpus, raw_tokens)
from textmining.recommender import ContentRecommender, precision_at_k  # noqa: E402
from textmining.tfidf import TfidfModel, cosine_pair_scalar  # noqa: E402

SEED = 42
SEEDS = [1, 7, 42, 123, 2024]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "report")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

TFIDF_KW = dict(min_df=2, max_df_ratio=0.9, max_features=5000, sublinear_tf=True)

# ------------------------------------------------------------------ plot style
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "axes.labelcolor": INK2, "axes.edgecolor": "#b9b8b2",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
FIGURES: list[str] = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(name)
    print("  figure:", name)


def r4(x):
    return round(float(x), 4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================== data
log("loading data")
docs, meta = load_default()
assert (docs["doc_id"] == meta["doc_id"]).all()
texts = docs["text"].tolist()
titles = meta["title"].tolist()
ids = meta["doc_id"].tolist()
labels = meta["category"].astype(str).tolist()
CATS = sorted(set(labels))
CCOL = {c: COLORS[i] for i, c in enumerate(CATS)}
N = len(texts)
R: dict = {"seed": SEED, "categories": CATS}

# ================================================================== baseline preprocessing + tf-idf
BASE = PreprocessConfig()
log("preprocessing (baseline)")
t0 = time.perf_counter()
toks = preprocess_corpus(texts, BASE)
t_pre = time.perf_counter() - t0
t0 = time.perf_counter()
tf = TfidfModel(**TFIDF_KW).fit(toks)
t_tfidf = time.perf_counter() - t0
X = tf.matrix

# ================================================================== 1. dataset statistics
log("dataset stats")
n_chars = [len(t) for t in texts]
n_words = [len(t.split()) for t in texts]
per_cat = Counter(labels)
tot_chars = sum(n_chars) + sum(len(t) for t in titles)
noise = {
    "docs_with_digits": int(sum(any(ch.isdigit() for ch in t) for t in texts)),
    "docs_with_currency_symbols": int(sum(("$" in t) or ("£" in t) for t in texts)),
    "docs_with_mojibake_A_circ": int(sum("Â" in t for t in texts)),
    "docs_with_quotes": int(sum('"' in t for t in texts)),
    "uppercase_token_share": r4(sum(sum(w[:1].isupper() for w in t.split()) for t in texts) / max(sum(n_words), 1)),
}
R["dataset"] = {
    "n_docs": N, "n_categories": len(CATS),
    "docs_per_category": {c: per_cat[c] for c in CATS},
    "total_chars": tot_chars, "total_mb": round(tot_chars / 1e6, 2),
    "words_total": sum(n_words),
    "chars": {"mean": round(np.mean(n_chars), 1), "median": float(np.median(n_chars)),
              "min": int(min(n_chars)), "max": int(max(n_chars)), "std": round(np.std(n_chars), 1)},
    "words": {"mean": round(np.mean(n_words), 1), "median": float(np.median(n_words)),
              "min": int(min(n_words)), "max": int(max(n_words)), "std": round(np.std(n_words), 1)},
    "words_per_category_mean": {c: round(float(np.mean([w for w, l in zip(n_words, labels) if l == c])), 1) for c in CATS},
    "metadata_columns": list(meta.columns), "document_columns": list(docs.columns),
    "noise": noise,
    "example_raw_snippet": texts[0][:260],
}

fig, ax = plt.subplots(figsize=(6.2, 2.8))
bars = ax.bar(CATS, [per_cat[c] for c in CATS], color=[CCOL[c] for c in CATS], width=0.6)
ax.bar_label(bars, padding=2, fontsize=8, color=INK)
ax.set_ylabel("documents"); ax.set_title("Documents per category (N = %d)" % N)
ax.grid(axis="x", visible=False)
save(fig, "01_category_distribution.png")

# ---------- duplicates
log("duplicate analysis")
dup = duplicate_report(texts, X, 0.9)
pairs = dup["pairs"]
exact_groups = dup["exact_groups"]
n_exact_pairs = sum(1 for p in pairs if p["type"] == "exact")
near_only = [p for p in pairs if p["type"] == "near"]
exact_docs_extra = sum(len(g) - 1 for g in exact_groups)
cross_cat_pairs = [p for p in pairs if labels[p["i"]] != labels[p["j"]]]
group_sizes = Counter(len(g) for g in exact_groups)
# exact pairs not caught by cosine (should be none) - each exact group is cosine 1.0
exact_pair_total = sum(len(g) * (len(g) - 1) // 2 for g in exact_groups)


def pair_row(p):
    return {"doc_a": ids[p["i"]], "doc_b": ids[p["j"]], "title_a": titles[p["i"]], "title_b": titles[p["j"]],
            "cat_a": labels[p["i"]], "cat_b": labels[p["j"]], "cosine": r4(p["cosine"]),
            "jaccard_5shingle": r4(p["jaccard_5shingle"]), "type": p["type"]}


near_sorted = sorted(near_only, key=lambda p: p["cosine"])
ex_rows = [pair_row(p) for p in pairs if p["type"] == "exact"][:4]
near_rows = [pair_row(p) for p in near_sorted[:3]] + [pair_row(p) for p in near_sorted[len(near_sorted) // 2:len(near_sorted) // 2 + 2]]
drop_by_cat = Counter(labels[i] for i in dup["drop_indices"])
R["duplicates"] = {
    "method": "exact = MD5 of lower-cased, whitespace-collapsed text; near = cosine(TF-IDF) >= 0.9, cross-checked with word 5-shingle Jaccard",
    "exact_groups": len(exact_groups), "exact_group_sizes": {str(k): v for k, v in sorted(group_sizes.items())},
    "docs_in_exact_groups": sum(len(g) for g in exact_groups), "redundant_exact_copies": exact_docs_extra,
    "exact_pairs": exact_pair_total,
    "pairs_cos_ge_0_9": len(pairs), "pairs_exact_type": n_exact_pairs, "pairs_near_only": len(near_only),
    "near_only_jaccard_mean": r4(np.mean([p["jaccard_5shingle"] for p in near_only])) if near_only else None,
    "near_only_jaccard_min": r4(min(p["jaccard_5shingle"] for p in near_only)) if near_only else None,
    "near_only_cos_min": r4(min(p["cosine"] for p in near_only)) if near_only else None,
    "cross_category_pairs": len(cross_cat_pairs),
    "cross_category_examples": [pair_row(p) for p in cross_cat_pairs[:3]],
    "docs_dropped_by_dedup": len(dup["drop_indices"]),
    "dropped_by_category": {c: drop_by_cat.get(c, 0) for c in CATS},
    "docs_after_dedup": N - len(dup["drop_indices"]),
    "examples_exact": ex_rows, "examples_near": near_rows,
}
fig, ax = plt.subplots(figsize=(6.2, 2.8))
if pairs:
    cs = [p["cosine"] for p in pairs]; js = [p["jaccard_5shingle"] for p in pairs]
    ex = [p["type"] == "exact" for p in pairs]
    ax.scatter([c for c, e in zip(cs, ex) if e], [j for j, e in zip(js, ex) if e], s=14, color=COLORS[0],
               label="exact duplicate (MD5 match)", alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.scatter([c for c, e in zip(cs, ex) if not e], [j for j, e in zip(js, ex) if not e], s=18, color=COLORS[1],
               marker="s", label="near duplicate only", alpha=0.9, edgecolor="white", linewidth=0.4)
ax.set_xlabel("cosine similarity of TF-IDF vectors"); ax.set_ylabel("5-shingle Jaccard")
ax.set_title("Duplicate pairs with cosine >= 0.9 (%d pairs)" % len(pairs)); ax.legend(loc="lower right")
save(fig, "02_duplicate_pairs.png")

# ================================================================== 2. preprocessing before / after
log("preprocessing before/after")
raw_toks = [raw_tokens(t) for t in texts]
st_raw = corpus_stats(raw_toks)
st_base = corpus_stats(toks)
lower_only = PreprocessConfig(remove_stopwords=False, normalizer="none", remove_numbers=False, min_token_len=1)
st_lower = corpus_stats(preprocess_corpus(texts, lower_only))


def slim(st):
    return {k: v for k, v in st.items() if k not in ("term_freq", "lengths")}


R["preprocessing"] = {
    "baseline_config": BASE.as_dict(), "baseline_label": BASE.label(),
    "raw": slim(st_raw), "lower_punct_only": slim(st_lower), "baseline": slim(st_base),
    "vocab_reduction_pct": round(100 * (1 - st_base["vocabulary_size"] / st_raw["vocabulary_size"]), 1),
    "token_reduction_pct": round(100 * (1 - st_base["total_tokens"] / st_raw["total_tokens"]), 1),
    "top20_raw": st_raw["term_freq"].most_common(20), "top20_baseline": st_base["term_freq"].most_common(20),
    "example": {"doc_id": ids[0], "raw_first_tokens": raw_toks[0][:30], "processed_first_tokens": toks[0][:22]},
    "time_s": round(t_pre, 2),
}
# cumulative effect of each preprocessing step on the vocabulary
STEP_CFGS = [
    ("1. Lowercase + drop punctuation", PreprocessConfig(remove_stopwords=False, normalizer="none", remove_numbers=False, min_token_len=1)),
    ("2. + drop numbers, min length 2", PreprocessConfig(remove_stopwords=False, normalizer="none")),
    ("3. + stop-word removal", PreprocessConfig(normalizer="none")),
    ("4. + Porter stemming (baseline)", BASE),
]
steps = [{"step": "0. Raw tokens (no normalisation)", **{k: v for k, v in slim(st_raw).items() if k != "documents"}}]
for nm, cfg_ in STEP_CFGS:
    st_ = st_base if cfg_ is BASE else corpus_stats(preprocess_corpus(texts, cfg_))
    steps.append({"step": nm, **{k: v for k, v in slim(st_).items() if k != "documents"}})
R["preprocessing"]["steps"] = steps

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9))
for ax, st, ttl, col in [(axes[0], st_raw, "Before: raw tokens", COLORS[1]),
                         (axes[1], st_base, "After: baseline pipeline", COLORS[0])]:
    top = st["term_freq"].most_common(20)[::-1]
    ax.barh([t for t, _ in top], [c for _, c in top], color=col, height=0.65)
    ax.set_title(ttl); ax.set_xlabel("corpus frequency"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
save(fig, "03_top20_before_after.png")

fig, ax = plt.subplots(figsize=(6.2, 3.2))
for st, lab, col in [(st_raw, "raw tokens", COLORS[1]), (st_base, "after preprocessing", COLORS[0])]:
    f = np.array(sorted(st["term_freq"].values(), reverse=True))
    ax.loglog(np.arange(1, len(f) + 1), f, color=col, lw=2, label=f"{lab} (|V| = {len(f):,})")
ranks = np.arange(1, 20001); c0 = sorted(st_raw["term_freq"].values(), reverse=True)[0]
ax.loglog(ranks, c0 / ranks, ls="--", color=INK2, lw=1, label="Zipf reference  f = C / rank")
ax.set_xlabel("rank (log)"); ax.set_ylabel("frequency (log)"); ax.set_title("Zipf plot: term frequency vs rank")
ax.legend()
save(fig, "04_zipf.png")

fig, ax = plt.subplots(figsize=(6.2, 3.0))
ax.hist(n_words, bins=60, color=COLORS[1], alpha=0.75, label="raw words / doc")
ax.hist(st_base["lengths"], bins=60, color=COLORS[0], alpha=0.75, label="tokens / doc after preprocessing")
ax.set_xlabel("length"); ax.set_ylabel("documents"); ax.set_title("Document length distribution"); ax.legend()
ax.set_xlim(0, np.percentile(n_words, 99.5))
save(fig, "05_doc_length_hist.png")

# ================================================================== 3. TF-IDF
log("tf-idf analysis")
idf_vals = np.array([tf.idf[t] for t in tf.vocab])
ex_term = "elect" if "elect" in tf.term_to_col else tf.vocab[len(tf.vocab) // 2]
ex_df = tf.index.df(ex_term)
d0 = 0
row_raw = Counter(toks[d0])
# manual walk-through for the top term of doc 0
weights_unnorm = {t: (1 + math.log10(c)) * tf.idf[t] for t, c in row_raw.items() if t in tf.term_to_col}
length0 = math.sqrt(sum(w * w for w in weights_unnorm.values()))
top_t = max(weights_unnorm, key=weights_unnorm.get)
walk = {"doc_id": ids[d0], "term": top_t, "tf": row_raw[top_t], "tf_weight": r4(1 + math.log10(row_raw[top_t])),
        "df": tf.index.df(top_t), "idf": r4(tf.idf[top_t]), "w_unnormalised": r4(weights_unnorm[top_t]),
        "doc_vector_length": r4(length0), "w_normalised": r4(weights_unnorm[top_t] / length0),
        "w_in_matrix": r4(X[d0, tf.term_to_col[top_t]])}
# cosine examples
S0 = X @ X[d0]
S0[d0] = -1
nn = int(np.argmax(S0))
other = next(i for i in range(N) if labels[i] != labels[d0] and i > 1000)
u = {tf.vocab[i]: float(X[d0, i]) for i in np.nonzero(X[d0])[0]}
v_nn = {tf.vocab[i]: float(X[nn, i]) for i in np.nonzero(X[nn])[0]}
v_ot = {tf.vocab[i]: float(X[other, i]) for i in np.nonzero(X[other])[0]}
R["tfidf"] = {
    "params": {"min_df": 2, "max_df_ratio": 0.9, "max_features": 5000, "tf": "1+log10(tf)", "idf": "log10(N/df)",
               "normalisation": "L2 (cosine)"},
    "pruning": tf.pruned_stats, "matrix_shape": list(X.shape),
    "density_pct": round(100 * float((X > 0).sum()) / X.size, 3),
    "avg_nonzero_per_doc": round(float((X > 0).sum(axis=1).mean()), 1),
    "row_norm_check": {"min": r4(np.linalg.norm(X, axis=1).min()), "max": r4(np.linalg.norm(X, axis=1).max())},
    "idf_stats": {"min": r4(idf_vals.min()), "max": r4(idf_vals.max()), "mean": r4(idf_vals.mean())},
    "lowest_idf_terms": [(t, tf.index.df(t), r4(tf.idf[t])) for t in sorted(tf.vocab, key=lambda t: tf.idf[t])[:8]],
    "highest_idf_terms_sample": [(t, tf.index.df(t), r4(tf.idf[t])) for t in sorted(tf.vocab, key=lambda t: -tf.idf[t])[:5]],
    "example_term": {"term": ex_term, "df": ex_df, "idf": r4(tf.idf.get(ex_term, 0)), "N": N,
                     "postings_sample": [(ids[d], c) for d, c in list(tf.index.lookup(ex_term).items())[:6]]},
    "walkthrough": walk,
    "example_doc_top_terms": {"doc_id": ids[d0], "title": titles[d0], "terms": [(t, r4(w)) for t, w in tf.top_terms(d0, 10)]},
    "cosine_examples": [
        {"a": ids[d0], "b": ids[nn], "title_a": titles[d0], "title_b": titles[nn], "cat_a": labels[d0], "cat_b": labels[nn],
         "cosine_matrix": r4(X[d0] @ X[nn]), "cosine_scalar": r4(cosine_pair_scalar(u, v_nn)), "note": "nearest neighbour"},
        {"a": ids[d0], "b": ids[other], "title_a": titles[d0], "title_b": titles[other], "cat_a": labels[d0], "cat_b": labels[other],
         "cosine_matrix": r4(X[d0] @ X[other]), "cosine_scalar": r4(cosine_pair_scalar(u, v_ot)), "note": "different category"},
    ],
    "time_s": round(t_tfidf, 2),
}
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.hist(idf_vals, bins=40, color=COLORS[0])
ax.set_xlabel("idf = log10(N / df)"); ax.set_ylabel("terms"); ax.set_title("IDF distribution of the 5,000 retained terms")
save(fig, "06_idf_hist.png")

# ================================================================== 4. Naive Bayes
log("naive bayes")
tr, te = stratified_split(labels, 0.2, SEED)
t0 = time.perf_counter()
nb = MultinomialNaiveBayes(1.0).fit([toks[i] for i in tr], [labels[i] for i in tr])
pred = nb.predict([toks[i] for i in te])
t_nb = time.perf_counter() - t0
y_te = [labels[i] for i in te]
rep = M.classification_report(y_te, pred, CATS)
probs = [nb.predict_proba_one(toks[i]) for i in te]
pred_rows = [{"doc_id": ids[i], "title": titles[i], "actual": y, "predicted": p, "posterior": r4(pr[p]),
              "correct": y == p} for i, y, p, pr in zip(te, y_te, pred, probs)]
pd.DataFrame(pred_rows).to_csv(os.path.join(OUT, "nb_test_predictions.csv"), index=False)
wrong = [r for r in pred_rows if not r["correct"]]
right = [r for r in pred_rows if r["correct"]]
rng = random.Random(SEED)
sample15 = sorted(wrong[:6] + rng.sample(right, 15 - min(6, len(wrong))), key=lambda r: r["doc_id"])
leak = leakage(tr, te, texts)
_nrm = lambda t: " ".join(str(t).lower().split())
_trset = {_nrm(texts[i]) for i in tr}
_leaked = [k for k, i in enumerate(te) if _nrm(texts[i]) in _trset]
_clean = [k for k in range(len(te)) if k not in set(_leaked)]
leak_acc = {"leaked_docs": len(_leaked),
            "acc_on_leaked": r4(np.mean([y_te[k] == pred[k] for k in _leaked])) if _leaked else None,
            "acc_on_non_leaked": r4(np.mean([y_te[k] == pred[k] for k in _clean])),
            "leaked_by_class": dict(Counter(y_te[k] for k in _leaked))}
R["naive_bayes"] = {
    "split": {"train": len(tr), "test": len(te), "test_size": 0.2, "seed": SEED,
              "train_per_class": dict(Counter(labels[i] for i in tr)), "test_per_class": dict(Counter(y_te))},
    "alpha": 1.0, "vocab_train": len(nb.vocab),
    "accuracy": r4(rep["accuracy"]),
    "per_class": {c: {k: (r4(v) if isinstance(v, float) else v) for k, v in d.items()} for c, d in rep["per_class"].items()},
    "macro": {k: r4(v) for k, v in rep["macro"].items()}, "weighted": {k: r4(v) for k, v in rep["weighted"].items()},
    "confusion_matrix": rep["confusion_matrix"], "labels": rep["labels"],
    "n_correct": len(right), "n_wrong": len(wrong),
    "misclassified": wrong, "sample_predictions": sample15,
    "actual_distribution": dict(Counter(y_te)), "predicted_distribution": dict(Counter(pred)),
    "top_terms": {c: [(t, round(s, 2)) for t, s in v[:10]] for c, v in nb.top_features(10).items()},
    "priors": {c: r4(math.exp(nb.log_prior[c])) for c in nb.classes},
    "test_leakage_docs": leak, "leakage_analysis": leak_acc, "time_s": round(t_nb, 2),
}
cm = np.array(rep["confusion_matrix"])
fig, ax = plt.subplots(figsize=(4.8, 4.0))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(CATS)), CATS, rotation=30, ha="right"); ax.set_yticks(range(len(CATS)), CATS)
for a in range(len(CATS)):
    for b in range(len(CATS)):
        ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=9, color="white" if cm[a, b] > cm.max() / 2 else INK)
ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
ax.set_title("Naive Bayes confusion matrix (test, n = %d)" % len(te))
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "07_nb_confusion_matrix.png")

fig, ax = plt.subplots(figsize=(6.2, 2.9))
w = 0.38; xs = np.arange(len(CATS))
a_ = [Counter(y_te)[c] for c in CATS]; p_ = [Counter(pred)[c] for c in CATS]
b1 = ax.bar(xs - w / 2, a_, w, color=COLORS[0], label="actual")
b2 = ax.bar(xs + w / 2, p_, w, color=COLORS[1], label="predicted")
ax.bar_label(b1, fontsize=7, padding=1); ax.bar_label(b2, fontsize=7, padding=1)
ax.set_xticks(xs, CATS); ax.set_ylabel("test documents"); ax.grid(axis="x", visible=False)
ax.set_title("Actual vs predicted class distribution (NB, test set)"); ax.legend(ncol=2, loc="upper right")
ax.set_ylim(0, max(a_ + p_) * 1.18)
save(fig, "08_nb_actual_vs_predicted.png")

fig, ax = plt.subplots(figsize=(6.2, 2.9))
w = 0.26
for k, (m, col) in enumerate([("precision", COLORS[0]), ("recall", COLORS[1]), ("f1", COLORS[2])]):
    ax.bar(xs + (k - 1) * w, [rep["per_class"][c][m] for c in CATS], w, color=col, label=m)
ax.set_xticks(xs, CATS); ax.set_ylim(0.8, 1.01); ax.set_ylabel("score"); ax.grid(axis="x", visible=False)
ax.set_title("NB per-class precision / recall / F1"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, "09_nb_per_class_metrics.png")

# ================================================================== 5. K-Means
log("k-means baseline")
t0 = time.perf_counter()
km = KMeans(k=5, init="kmeans++", n_init=5, seed=SEED, max_iter=100, tol=1e-4).fit(X)
t_km = time.perf_counter() - t0
cl = km.labels_.tolist()
maj = M.majority_label_map(cl, labels)
cont = M.contingency(cl, labels)
ric = M.rand_index_counts(cl, labels)
pur = M.purity(cl, labels)
rs = random.Random(SEED).sample(range(N), 300)
ri_fast_s = M.rand_index([cl[i] for i in rs], [labels[i] for i in rs])
ri_bf_s = M.rand_index_bruteforce([cl[i] for i in rs], [labels[i] for i in rs])
top_terms_cl = km.top_terms(tf.vocab, 10)
mem = pd.DataFrame({"doc_id": ids, "title": titles, "true_label": labels, "cluster": cl,
                    "cluster_majority": [maj[c] for c in cl]})
mem.to_csv(os.path.join(OUT, "kmeans_membership.csv"), index=False)
mem_sample = []
for c in range(5):
    sub = mem[mem.cluster == c]
    mis = sub[sub.true_label != sub.cluster_majority].head(1)
    ok = sub[sub.true_label == sub.cluster_majority].sample(2, random_state=SEED)
    mem_sample += pd.concat([ok, mis]).to_dict("records")
R["kmeans"] = {
    "params": {"k": 5, "init": "kmeans++", "n_init": 5, "max_iter": 100, "tol": 1e-4, "seed": SEED,
               "distance": "squared Euclidean on L2-normalised TF-IDF rows"},
    "init_docs": [{"cluster": j, "doc_id": ids[i], "title": titles[i], "category": labels[i]} for j, i in enumerate(km.init_indices)],
    "history": km.history, "stop_reason": km.stop_reason, "n_iter": km.n_iter_, "sse": r4(km.sse_),
    "restarts": km.run_summary,
    "cluster_sizes": {int(c): int(n) for c, n in sorted(Counter(cl).items())},
    "majority_label": {int(k): v for k, v in sorted(maj.items())},
    "contingency": {int(k): {c: int(v.get(c, 0)) for c in CATS} for k, v in sorted(cont.items())},
    "purity": r4(pur), "rand_index": r4(ric["rand_index"]),
    "pair_counts": {k: int(v) for k, v in ric.items() if k != "rand_index"},
    "bruteforce_check": {"sample": 300, "fast": r4(ri_fast_s), "bruteforce": r4(ri_bf_s),
                         "match": abs(ri_fast_s - ri_bf_s) < 1e-12},
    "top_terms": {int(k): v for k, v in top_terms_cl.items()},
    "membership_sample": mem_sample, "time_s": round(t_km, 2),
    "label_permutation_note": "cluster ids are arbitrary; majority label used only to name clusters",
}

hist = km.history
fig, ax1 = plt.subplots(1, 2, figsize=(7.4, 2.9))
it = [h["iteration"] for h in hist]
ax1[0].plot(it, [h["SSE"] for h in hist], color=COLORS[0], lw=2, marker="o", ms=3)
ax1[0].set_xlabel("iteration"); ax1[0].set_ylabel("SSE (RSS)"); ax1[0].set_title("SSE per iteration (best run)")
ax1[1].bar(it, [h["docs_changed"] for h in hist], color=COLORS[1], width=0.7)
ax1[1].set_yscale("symlog"); ax1[1].set_xlabel("iteration"); ax1[1].set_ylabel("docs changing cluster")
ax1[1].set_title("Reassignments per iteration"); ax1[1].grid(axis="x", visible=False)
save(fig, "10_kmeans_sse_iterations.png")

log("pca projection")
P = pca_2d(X)
cname = {c: f"C{c} ({maj[c]})" for c in range(5)}
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharex=True, sharey=True)
for c in range(5):
    m = np.array(cl) == c
    axes[0].scatter(P[m, 0], P[m, 1], s=7, color=COLORS[c], marker=MARKERS[c], alpha=0.7, linewidths=0, label=cname[c])
for j, c in enumerate(CATS):
    m = np.array(labels) == c
    axes[1].scatter(P[m, 0], P[m, 1], s=7, color=CCOL[c], marker=MARKERS[j], alpha=0.7, linewidths=0, label=c)
axes[0].set_title("Coloured by K-Means cluster"); axes[1].set_title("Coloured by true category")
for ax in axes:
    ax.set_xlabel("PC 1"); ax.legend(fontsize=6.5, markerscale=2, loc="best")
axes[0].set_ylabel("PC 2")
save(fig, "11_kmeans_pca_scatter.png")

fig, ax = plt.subplots(figsize=(6.2, 2.8))
sz = [Counter(cl)[c] for c in range(5)]
bars = ax.bar([cname[c] for c in range(5)], sz, color=COLORS[:5], width=0.6)
ax.bar_label(bars, fontsize=8, padding=2); ax.set_ylabel("documents"); ax.grid(axis="x", visible=False)
ax.set_title("Documents per K-Means cluster (k = 5)")
save(fig, "12_kmeans_cluster_sizes.png")

ct = np.array([[cont[c].get(cat, 0) for cat in CATS] for c in range(5)])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.imshow(ct, cmap="Blues")
ax.set_xticks(range(len(CATS)), CATS, rotation=30, ha="right"); ax.set_yticks(range(5), [cname[c] for c in range(5)])
for a in range(5):
    for b in range(len(CATS)):
        ax.text(b, a, ct[a, b], ha="center", va="center", fontsize=8.5, color="white" if ct[a, b] > ct.max() / 2 else INK)
ax.grid(False); ax.set_xlabel("true category"); ax.set_title("Contingency table: cluster x category")
save(fig, "13_kmeans_contingency.png")

log("elbow")
el = elbow(X, range(2, 11), seed=SEED)
el_pur = []
for k_ in range(2, 11):
    kmk = KMeans(k=k_, seed=SEED, n_init=2, max_iter=50).fit(X)
    el_pur.append((k_, r4(M.purity(kmk.labels_.tolist(), labels)), r4(M.rand_index(kmk.labels_.tolist(), labels))))
R["kmeans"]["elbow"] = [(k_, r4(s)) for k_, s in el]
R["kmeans"]["purity_by_k"] = el_pur
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
axes[0].plot([k_ for k_, _ in el], [s for _, s in el], marker="o", color=COLORS[0], lw=2)
axes[0].axvline(5, color=INK2, ls="--", lw=1); axes[0].set_xlabel("k"); axes[0].set_ylabel("SSE")
axes[0].set_title("Elbow curve (SSE vs k)")
axes[1].plot([e[0] for e in el_pur], [e[1] for e in el_pur], marker="o", color=COLORS[1], lw=2, label="purity")
axes[1].plot([e[0] for e in el_pur], [e[2] for e in el_pur], marker="s", color=COLORS[2], lw=2, label="Rand index")
axes[1].axvline(5, color=INK2, ls="--", lw=1); axes[1].set_xlabel("k"); axes[1].set_title("Purity / RI vs k")
axes[1].legend()
save(fig, "14_kmeans_elbow.png")

# ================================================================== 6. seed stability
log("seed stability")
keep = [i for i in range(N) if i not in set(dup["drop_indices"])]
toks_d = [toks[i] for i in keep]; lab_d = [labels[i] for i in keep]
tf_d = TfidfModel(**TFIDF_KW).fit(toks_d)
stab = {}
for name, Xs, ls_ in [("full", X, labels), ("dedup", tf_d.matrix, lab_d)]:
    for ninit in (5, 1):
        runs = []
        for s in SEEDS:
            k_ = KMeans(k=5, seed=s, n_init=ninit).fit(Xs)
            runs.append({"seed": s, "purity": r4(M.purity(k_.labels_.tolist(), ls_)),
                         "rand_index": r4(M.rand_index(k_.labels_.tolist(), ls_)), "sse": r4(k_.sse_),
                         "iterations": k_.n_iter_})
        pu = [r["purity"] for r in runs]; ri = [r["rand_index"] for r in runs]
        stab[f"{name}_ninit{ninit}"] = {"runs": runs, "purity_mean": r4(np.mean(pu)), "purity_std": r4(np.std(pu)),
                                        "ri_mean": r4(np.mean(ri)), "ri_std": r4(np.std(ri)),
                                        "purity_min": r4(min(pu)), "purity_max": r4(max(pu))}
# NB over split seeds (full vs dedup)
nb_stab = {}
for name, T_, L_, TX in [("full", toks, labels, texts), ("dedup", toks_d, lab_d, [texts[i] for i in keep])]:
    accs, f1s, leaks = [], [], []
    for s in SEEDS:
        a, b = stratified_split(L_, 0.2, s)
        m_ = MultinomialNaiveBayes().fit([T_[i] for i in a], [L_[i] for i in a])
        rp = M.classification_report([L_[i] for i in b], m_.predict([T_[i] for i in b]), CATS)
        accs.append(rp["accuracy"]); f1s.append(rp["macro"]["f1"]); leaks.append(leakage(a, b, TX))
    nb_stab[name] = {"acc": [r4(a) for a in accs], "acc_mean": r4(np.mean(accs)), "acc_std": r4(np.std(accs)),
                     "f1_mean": r4(np.mean(f1s)), "f1_std": r4(np.std(f1s)), "leakage": leaks}
k20 = KMeans(k=5, seed=SEED, n_init=20).fit(tf_d.matrix)
stab["dedup_seed42_ninit20"] = {"purity": r4(M.purity(k20.labels_.tolist(), lab_d)),
                                "rand_index": r4(M.rand_index(k20.labels_.tolist(), lab_d)), "sse": r4(k20.sse_)}
R["stability"] = {"seeds": SEEDS, "kmeans": stab, "naive_bayes": nb_stab}

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
xs = np.arange(len(SEEDS))
for ax, key, ttl in [(axes[0], "ninit5", "K-Means purity, n_init = 5"), (axes[1], "ninit1", "K-Means purity, n_init = 1")]:
    for j, (nm, col) in enumerate([("full", COLORS[0]), ("dedup", COLORS[1])]):
        vals = [r["purity"] for r in stab[f"{nm}_{key}"]["runs"]]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=f"{nm} corpus")
    ax.set_xticks(xs, [str(s) for s in SEEDS]); ax.set_xlabel("seed"); ax.set_title(ttl)
    ax.set_ylim(0.4, 1.02); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("purity"); axes[0].legend(loc="lower left")
save(fig, "15_seed_stability.png")

# ================================================================== 7. comparison
fig, ax = plt.subplots(figsize=(6.6, 3.0))
names = ["NB accuracy", "NB macro-P", "NB macro-R", "NB macro-F1", "KM purity", "KM Rand index",
         "KM purity\n(5-seed mean)", "KM RI\n(5-seed mean)"]
vals = [rep["accuracy"], rep["macro"]["precision"], rep["macro"]["recall"], rep["macro"]["f1"], pur, ric["rand_index"],
        stab["full_ninit5"]["purity_mean"], stab["full_ninit5"]["ri_mean"]]
cols = [COLORS[0]] * 4 + [COLORS[1]] * 2 + [COLORS[2]] * 2
bars = ax.bar(names, vals, color=cols, width=0.62)
ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], fontsize=7, padding=1)
ax.set_ylim(0, 1.1); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", labelsize=7)
ax.set_title("Supervised NB (blue) vs unsupervised K-Means (orange: seed 42, green: mean of 5 seeds)")
save(fig, "16_nb_vs_kmeans.png")
R["comparison"] = dict(zip([n.replace("\n", " ") for n in names], [r4(v) for v in vals]))

# ================================================================== 8. recommendation
log("recommendation")
rec = ContentRecommender(tf, meta)
dup_partner = {}
for p in pairs:
    dup_partner.setdefault(p["i"], []).append(p["j"]); dup_partner.setdefault(p["j"], []).append(p["i"])


def first_of(cat, need_dup=False):
    for i in range(N):
        if labels[i] == cat and ((i in dup_partner) == need_dup):
            return i


queries = [first_of("business", True), first_of("politics"), first_of("entertainment"), first_of("tech")]


def rec_rows(q, k=5, thr=None):
    out = []
    for r, (j, s) in enumerate(rec.similar_to_doc(q, k, thr), 1):
        out.append({"rank": r, "doc_id": ids[j], "title": titles[j], "category": labels[j],
                    "source": meta.loc[j, "source"], "period": meta.loc[j, "period"], "n_words": int(meta.loc[j, "n_words"]),
                    "score": r4(s), "same_category": labels[j] == labels[q],
                    "shared_terms": [t for t, _ in rec.explain(X[q], j, 5)]})
    return out


rec_q = []
for q in queries:
    rec_q.append({"query": {"doc_id": ids[q], "title": titles[q], "category": labels[q]},
                  "top5": rec_rows(q), "top5_hide_dups": rec_rows(q, 5, 0.9)})
free_text = "mobile phone operators launch faster broadband and music download services"
qt = preprocess(free_text, BASE)
ft = [{"rank": r, "doc_id": ids[j], "title": titles[j], "category": labels[j], "score": r4(s)}
      for r, (j, s) in enumerate(rec.similar_to_text(qt, 5), 1)]
log("precision@k (full corpus, 4 passes)")
pk = {}
for k_ in (5, 10):
    pk[f"P@{k_}"] = r4(precision_at_k(rec, labels, k_))
    pk[f"P@{k_}_hide_dups"] = r4(precision_at_k(rec, labels, k_, dup_threshold=0.9))
rec_d = ContentRecommender(tf_d, meta.iloc[keep])
pk["P@5_dedup_corpus"] = r4(precision_at_k(rec_d, lab_d, 5))
pk["P@10_dedup_corpus"] = r4(precision_at_k(rec_d, lab_d, 10))
# how many top-5 lists contain at least one near-duplicate of the query
flood = sum(1 for i in range(N) if any(s >= 0.9 for _, s in rec.similar_to_doc(i, 5)))
# per-category P@5
pc = {}
for c in CATS:
    idx = [i for i in range(N) if labels[i] == c]
    pc[c] = r4(precision_at_k(rec, labels, 5, idx))
R["recommendation"] = {"queries": rec_q, "free_text": {"query": free_text, "tokens": qt, "results": ft},
                       "precision": pk, "p5_per_category": pc, "lists_with_dup_in_top5": flood,
                       "relevance_proxy": "a recommendation is relevant if it shares the query's category"}
fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.4))
for ax, qd in zip(axes.ravel(), rec_q):
    sc = [r["score"] for r in qd["top5"]][::-1]
    cs = [CCOL[r["category"]] for r in qd["top5"]][::-1]
    ax.barh([f"#{r['rank']} {r['doc_id']}" for r in qd["top5"]][::-1], sc, color=cs, height=0.6)
    ax.set_title(f"Query {qd['query']['doc_id']}", fontsize=9); ax.set_xlim(0, 1.05); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7); ax.set_xlabel("cosine")
save(fig, "17_recommendation_scores.png")

# ================================================================== 9. ablation
log("ablation (this is the slow part)")
CONFIGS = [
    ("Baseline", PreprocessConfig()),
    ("No stop-word removal", PreprocessConfig(remove_stopwords=False)),
    ("Lemmatization", PreprocessConfig(normalizer="lemma")),
    ("No normalizer", PreprocessConfig(normalizer="none")),
    ("Uni+bigram", PreprocessConfig(ngram_max=2)),
    ("No lowercasing", PreprocessConfig(lowercase=False)),
    ("No lowercasing, no stemming", PreprocessConfig(lowercase=False, normalizer="none")),
]
abl = []
for name, cfg in CONFIGS:
    t0 = time.perf_counter()
    tk = preprocess_corpus(texts, cfg)
    t_cfg = time.perf_counter() - t0
    for dd in (False, True):
        log(f"  {name} dedup={dd}")
        res = run_experiment(docs, meta, cfg, dedup=dd, tokens=tk, seed=SEED)
        # seed-averaged K-Means purity for a fairer comparison than a single seed
        keep_i = list(range(N))
        if dd:
            tmp = TfidfModel(**TFIDF_KW).fit(tk)
            drop = set(duplicate_report(texts, tmp.matrix, 0.9)["drop_indices"])
            keep_i = [i for i in keep_i if i not in drop]
        Xm = TfidfModel(**TFIDF_KW).fit([tk[i] for i in keep_i]).matrix
        lb = [labels[i] for i in keep_i]
        pus, ris = [], []
        for s in SEEDS:
            k_ = KMeans(k=5, seed=s, n_init=2).fit(Xm)
            pus.append(M.purity(k_.labels_.tolist(), lb)); ris.append(M.rand_index(k_.labels_.tolist(), lb))
        res = {k: (r4(v) if isinstance(v, float) else v) for k, v in res.items()}
        res["timings"]["preprocess_s"] = t_cfg
        res["timings"] = {k: round(v, 2) for k, v in res["timings"].items()}
        res.update({"name": name + (" + dedup" if dd else ""), "base_name": name,
                    "km_purity_seedmean": r4(np.mean(pus)), "km_purity_seedstd": r4(np.std(pus)),
                    "km_ri_seedmean": r4(np.mean(ris))})
        abl.append(res)
R["ablation"] = abl

fig, axes = plt.subplots(1, 4, figsize=(8.0, 3.3), sharey=True)
names_ = [n for n, _ in CONFIGS]
ys = np.arange(len(names_))
for ax, key, ttl, lim in [(axes[0], "vocab_raw", "raw vocabulary", None), (axes[1], "nb_accuracy", "NB accuracy", (0.9, 1.0)),
                          (axes[2], "km_purity_seedmean", "KM purity (5-seed mean)", (0.4, 1.0)),
                          (axes[3], "precision_at_5", "P@5", (0.85, 1.0))]:
    for j, (dd, col) in enumerate([(False, COLORS[0]), (True, COLORS[1])]):
        v = [next(a[key] for a in abl if a["base_name"] == n and a["dedup"] == dd) for n in names_]
        ax.barh(ys + (j - 0.5) * 0.38, v, 0.38, color=col, label="with duplicates" if not dd else "deduplicated")
    ax.set_title(ttl, fontsize=9); ax.grid(axis="y", visible=False)
    if lim:
        ax.set_xlim(*lim)
    ax.tick_params(axis="x", labelsize=7)
axes[0].set_yticks(ys, names_); axes[0].invert_yaxis(); axes[0].tick_params(axis="y", labelsize=7.5)
axes[0].set_xscale("log"); axes[0].set_xlim(1e4, 5e5); axes[0].set_title("raw vocabulary (log)", fontsize=9)
axes[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
axes[1].set_xticks([0.90, 0.95, 1.0]); axes[2].set_xticks([0.4, 0.6, 0.8, 1.0]); axes[3].set_xticks([0.85, 0.90, 0.95, 1.0])
h, l_ = axes[3].get_legend_handles_labels()
fig.legend(h, l_, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
save(fig, "18_ablation.png")

# ================================================================== write
R["timings_s"] = {"preprocess": round(t_pre, 2), "tfidf": round(t_tfidf, 2), "naive_bayes": round(t_nb, 2),
                  "kmeans": round(t_km, 2)}
R["figures"] = FIGURES
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log(f"done -> report/results.json, {len(FIGURES)} figures")
