# 2. Architecture and Implementation

## 2.1 Overview

```
                    ┌───────────────────────── Streamlit UI (app.py + ui/) ─────────────────────────┐
                    │ Dashboard · Ingestion · Preprocessing · Mining Hub · Recommendation ·         │
                    │ Performance · Inferences                                                      │
                    └───────────────┬───────────────────────────────────────────────────────────────┘
                                    │  ui/state.py  — lazy Pipeline object, every step cached with
                                    │                 st.cache_resource keyed on (dataset, config JSON)
                                    ▼
 data/*.csv / upload ──► data_loader ──► preprocessing ──► tfidf (inverted index, TF-IDF, cosine)
                                                              │
                     ┌───────────────┬──────────────────────┼────────────────┬──────────────────┐
                     ▼               ▼                      ▼                ▼                  ▼
               duplicates      naive_bayes              kmeans           recommender          pca (plot only)
                     │               │                      │                │
                     └───────────────┴────► metrics (P/R/F1, confusion, Purity, Rand Index) ◄───┘

 experiments.py ── runs the same textmining/ code ──► report/results.json + figures ──► report/build_report.py ──► Report.pdf
```

The **`textmining/` package** holds all the algorithms and does not depend on Streamlit. The app, the
experiments script and the tests all call the same code, so the numbers in the app, the report and the tests
match.

## 2.2 Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit entry point: page config, CSS, sidebar navigation (`st.sidebar.radio`, key `nav`) |
| `ui/state.py` | `Pipeline`: lazy properties (tokens → TF-IDF → dups → NB → K-Means → PCA → recommender), all cached |
| `ui/common.py` | Shared Plotly template and category colours |
| `ui/page_*.py` | One file per page |
| `textmining/data_loader.py` | Default corpus loader; CSV/JSON/JSONL/ZIP upload; text vs metadata split; unique IDs |
| `textmining/preprocessing.py` | `PreprocessConfig` + `preprocess()`; `corpus_stats()`; offline fallbacks |
| `textmining/tfidf.py` | `InvertedIndex`, `TfidfModel`, `cosine_similarity`, `cosine_pair_scalar` |
| `textmining/naive_bayes.py` | `MultinomialNaiveBayes`, `stratified_split` |
| `textmining/kmeans.py` | `KMeans` with a full step log, `elbow()` |
| `textmining/metrics.py` | Confusion matrix, P/R/F1 (macro, weighted), Purity, Rand Index (fast + brute force) |
| `textmining/duplicates.py` | Exact (MD5) and near (cosine + 5-shingle Jaccard) duplicates; union-find drop list |
| `textmining/recommender.py` | `ContentRecommender` (Top-K by document or free text, explanation), `precision_at_k` |
| `textmining/pca.py` | 2-D PCA via the n×n Gram matrix (visualisation only) |
| `textmining/pipeline.py` | `run_experiment()`, one end-to-end run used by the ablation; `leakage()` |
| `experiments.py` | Reproducible experiments (seed 42) → `report/results.json`, `report/figures/` |
| `report/build_report.py` | Builds `report/Report.pdf` from the results |
| `data/prepare_dataset.py` | Raw `bbc/<category>/<nnn>.txt` → `documents.csv` + `metadata.csv` |
| `tests/` | pytest suite: unit, integration, end-to-end, compliance |

## 2.3 Algorithms

Everything below is plain Python plus numpy array arithmetic.

### Preprocessing (`preprocessing.py`)
Order: HTML-unescape, strip tags and URLs → lowercase → regex tokenise → drop pure-punctuation tokens →
drop tokens containing digits → minimum length → stop-words (NLTK list + `said`, `mr`, …) → Porter stem **or**
WordNet lemma → n-grams (joined with `_`, e.g. `prime_minist`). Each step is a field of `PreprocessConfig`.
If NLTK data can't be downloaded (offline lab), a built-in stop-word list and a rule-based lemmatiser are used
instead.

### Inverted index and TF-IDF (`tfidf.py`)
* `InvertedIndex.build()` maps each term to `{doc_idx: raw tf}`, so df is the postings length.
* Vocabulary pruning: `min_df=2`, `max_df=0.9·N`, keep the top `max_features=5000` by collection frequency.
* Weighting (SMART **ltc**), filled term-at-a-time from the postings lists:

  `w(t,d) = (1 + log10 tf(t,d)) · log10(N / df(t))`, then `w'(t,d) = w(t,d) / sqrt(Σ_t w(t,d)²)`

* Cosine similarity: `cos(a,b) = a·b / (|a||b|)`. Rows are already unit length, so this reduces to a dot product.
  `cosine_pair_scalar` is a dictionary-based version used to cross-check it.

### Multinomial Naive Bayes (`naive_bayes.py`)
* Manual stratified 80/20 split (seed 42). The **vocabulary comes from the training set only**.
* `P(c) = N_c / N`; `P(t|c) = (T_ct + α) / (Σ_t' T_ct' + α|V|)` with Laplace α = 1.
* Prediction: `argmax_c [log P(c) + Σ_t tf_t · log P(t|c)]`, summed in log space to avoid underflow; terms not
  in the vocabulary are ignored.
* Posteriors are normalised with log-sum-exp. "Top indicative terms" ranks each term by
  `log P(t|c) − mean_{c'} log P(t|c')`.

### K-Means (`kmeans.py`)
1. **Initialisation:** `kmeans++` (D² sampling) or `random` (k distinct documents). The chosen seed documents are recorded.
2. **Assignment:** each document goes to the nearest centroid by squared Euclidean distance (`||x||² − 2x·c + ||c||²`).
3. **Update:** each centroid becomes the mean of its members. An empty cluster is re-seeded with the farthest point.
4. **Stopping:** stop when no document changes cluster, or the largest centroid shift is below `tol`, or after `max_iter` iterations.
5. Repeat for `n_init` restarts and keep the run with the lowest SSE. Every iteration logs documents changed, centroid shift, SSE and cluster sizes.

### Evaluation (`metrics.py`)
* **Purity** = `(1/N) Σ_k max_j |ω_k ∩ c_j|`
* **Rand Index** = `(TP + TN) / C(N,2)`, computed exactly from the contingency table:
  TP = Σ C(n_kj, 2), TP+FP = Σ_k C(|ω_k|, 2), TP+FN = Σ_j C(|c_j|, 2).
  `rand_index_bruteforce` loops over all pairs, and the two versions are checked to agree.
* Classification: confusion matrix (rows = true labels), per-class P/R/F1, macro and weighted averages, accuracy.

### Duplicates (`duplicates.py`)
* Exact: MD5 hash of the lowercased, whitespace-collapsed text.
* Near: pairs with TF-IDF cosine ≥ threshold (default 0.9), cross-checked with 5-word-shingle Jaccard.
* All pairs are merged with union-find. The first member of each group is kept and the rest go in
  `drop_indices`, which feeds the "Remove duplicates before modelling" switch.

### Recommendation (`recommender.py`)
* Score = cosine(query vector, document vector). The query is a corpus document or free text run through the same preprocessing and IDF.
* The query itself is excluded. If enabled, any candidate with cosine ≥ the duplicate threshold is skipped as a near-copy.
* The explanation shows the top terms by contribution `q_t · d_t` to the dot product.
* Evaluation: Precision@K with "same category as the query" as a stand-in for relevance.

## 2.4 Design decisions

| Decision | Reason |
|---|---|
| Dense float32 TF-IDF matrix capped at 5,000 features | 2,225 × 5,000 × 4 B ≈ 44 MB, which makes K-Means and cosine fast in numpy; the cap removes rare-term noise |
| NB uses raw term counts, not TF-IDF | The multinomial model assumes counts (IIR ch. 13) |
| TF-IDF is fitted on the full corpus | Used only by the unsupervised steps (clustering, recommendation, duplicates). NB has its own train-only vocabulary, so no test information leaks into the classifier |
| PCA only for plotting | Clustering always runs on the full 5,000-dimension TF-IDF space |
| Multiple K-Means restarts and seeds | One seed can land in a poor local optimum (purity 0.79 on the de-duplicated corpus with seed 42), so results are reported as a 5-seed mean ± std |
| Every Streamlit step cached by config | Changing one setting reruns only what depends on it; pages render in under ~1.5 s after the first load |
| Text and metadata in separate files | Required by the brief; also mirrors a real document store |
