# 1. The Assignment and How We Solved It

## 1.1 What the assignment asks

**Course:** Information Retrieval (AIMLCZG537 / DSECLZG537), S1-26 — Assignment 2 · 10 marks · due 9 Nov 2026, 23:59.

You act as a text mining / NLP engineer. Pick a realistic domain, take a real document collection, and build
an **end-to-end Streamlit application** that:

1. ingests and profiles the corpus,
2. preprocesses the text with a configurable pipeline,
3. builds an **inverted index and TF-IDF features by hand**, with cosine similarity,
4. **classifies** documents with a hand-written **Naive Bayes**,
5. **clusters** them with a hand-written **K-Means**, showing each step,
6. evaluates clustering with hand-written **Purity** and **Rand Index**, and compares it with classification,
7. **recommends** the Top-K related documents,
8. visualises all of the above in the app, and
9. is backed by a PDF report that answers **five compulsory discussion questions** with experimental evidence.

Built-in library implementations of TF-IDF, Naive Bayes, K-Means, Purity and Rand Index are **prohibited**.

### Rubric

| Component | Marks |
|---|---|
| Streamlit workflow and user interface | 1 |
| Text preprocessing and feature engineering | 1 |
| Text mining: manual classification, clustering, evaluation | 3 |
| Document recommendation module | 1 |
| Visualisations of classification, clustering, recommendation | 1 |
| Experimental results, compulsory inferences, comparison, technical report | 2 |
| Execution and demonstration in the BITS Virtual Lab | 1 |

## 1.2 What we chose

| Decision | Choice | Why |
|---|---|---|
| Business context | News intelligence for a media-monitoring / business-research firm | Analysts must route articles to desks, find themes, drop syndicated copies and find related coverage, so all four techniques solve a real need |
| Dataset | BBC News full-text corpus (Greene & Cunningham, ICML 2006) | Real, public and labelled: 2,225 articles, 5 categories, about 5 MB, with real duplicates |
| Recommendation | Content-based (TF-IDF + cosine) | No user–document interaction log exists for this corpus. Collaborative filtering is discussed in the report but not faked |
| Language / UI | Python 3 + Streamlit + Plotly | Required by the assignment; interactive charts |
| Algorithms | Written from scratch in `textmining/`. **scikit-learn is not installed** | Satisfies the "no built-in implementation" rule |

## 1.3 Requirement traceability

Each line of the assignment brief is listed below with where it is implemented, where it appears in the app,
and where the report covers it.

### A. Streamlit workflow

| Requirement | Code | App page |
|---|---|---|
| Dashboard and corpus overview | `ui/page_dashboard.py` | Dashboard |
| Data upload / ingestion | `textmining/data_loader.py`, `ui/page_ingestion.py` | Data Ingestion |
| Preprocessing configuration | `ui/page_preprocess.py` | Preprocessing & Features |
| Text mining & corpus analytics hub | `ui/page_mining.py` | Text Mining Hub (4 tabs) |
| Recommendation panel | `ui/page_recommend.py` | Recommendation |
| Performance analytics | `ui/page_performance.py` | Performance Analytics |
| *(extra)* compulsory inferences with live numbers | `ui/page_inferences.py` | Inferences & Discussion |

### B. Dataset and ingestion

| Requirement | How it is met |
|---|---|
| Authentic dataset, CSV/JSON/folder of text | Real BBC articles, shipped as `data/*.csv` plus the original `bbc-fulltext.zip`. Uploads of CSV, JSON, JSONL or a ZIP of `.txt` files are also accepted |
| Unique doc ID + main text field | `doc_id` (e.g. `BUS-001`) + `text`. The loader de-duplicates IDs on upload |
| Category labels + metadata | `category` (5 classes); `title`, `source`, `period`, `filename`, `n_chars`, `n_words` |
| ≥ 50–75 docs or ~5 MB | 2,225 docs, 5.03 M characters |
| Realistic noise | Mixed case, quotes, numbers, `£`/`$`, proper nouns, jargon. The Ingestion page shows examples |
| Duplicates / near-duplicates | 137 exact-duplicate pairs + 26 near-duplicate pairs (cosine ≥ 0.9). Detected in `textmining/duplicates.py` |
| Text and metadata separate | `documents.csv` (doc_id, text) and `metadata.csv` (everything else) |
| Corpus statistics after ingestion | Data Ingestion page: counts per category, length statistics, unique-ID check, missing values |

### C. Preprocessing and features

| Requirement | Code |
|---|---|
| Lowercasing, tokenisation, stop-words, stemming, lemmatisation, n-grams | `textmining/preprocessing.py` → `PreprocessConfig`, `preprocess()` (each step is a toggle) |
| **Inverted index built by hand** | `textmining/tfidf.py` → `InvertedIndex` (term → {doc: tf}) |
| **TF-IDF with length normalisation, by hand** | `TfidfModel.fit()`: `w = (1+log10 tf)·log10(N/df)`, then each row is divided by `sqrt(Σw²)` |
| Cosine similarity | `cosine_similarity()` and the pure-Python `cosine_pair_scalar()` |
| Vocabulary / TF distributions before and after | Preprocessing page → "Before vs after" tab (stats table, top-20 bars, Zipf plot, length histograms) |
| Corpus / feature visualisations | IDF histogram, inverted-index explorer, cosine heatmap, duplicate table |

### D. Text mining and analytics

| Requirement | Code | Where shown |
|---|---|---|
| Naive Bayes by hand, train/test split, predictions, Acc/P/R/F1 | `naive_bayes.py` (multinomial, Laplace), `stratified_split()`; `metrics.classification_report()` | Mining Hub → Naive Bayes |
| K-Means by hand: init, assignment, update, stopping shown | `kmeans.py` → `history`, `init_indices`, `stop_reason`, `run_summary` | Mining Hub → K-Means |
| Purity and Rand Index by hand | `metrics.purity()`, `metrics.rand_index_counts()` (+ brute-force check) | Mining Hub → Evaluation & comparison |
| NB vs K-Means comparison | `ui/page_mining.py::interpretation()` | Evaluation & comparison tab |
| Actual vs predicted chart, confusion matrix, metrics | Plotly | Naive Bayes tab |
| 2-D cluster scatter (PCA for display only) | `pca.py` (numpy eigen-decomposition) | K-Means tab |
| Cluster-size bar chart, membership table | — | K-Means tab |
| Purity/RI next to NB metrics + interpretation | — | Evaluation & comparison tab |

### E. Recommendation

| Requirement | How it is met |
|---|---|
| Content-based / collaborative / hybrid | Content-based vector-space model (`recommender.py`) |
| Select or enter a document → Top-K | Choose a document from a searchable list, or type free text, then pick K with the slider |
| Scores + metadata | Rank, cosine score, doc_id, title, category, source, period, top shared terms |
| Explain method + limitations | "How it works", "Limitations" and "Content-based vs collaborative" expanders; report §8 |

### F. Inferences (compulsory)

All five questions are answered in **report §9**, using numbers from the ablation, de-duplication and seed
experiments. The app's **Inferences & Discussion** page answers them again with live numbers.

### Submission components

| Item | File |
|---|---|
| Streamlit source code | `app.py`, `ui/`, `textmining/` |
| Dataset | `data/documents.csv`, `data/metadata.csv`, `data/bbc-fulltext.zip` (in the zip) |
| Report (PDF) | `report/Report.pdf` |
| Virtual Lab evidence | `screenshots/` + screen recording (**student action**) |
| README | `README.md` |
| Contribution sheet | `G35_Contribution.xlsx` |
| Zip | `./make_zip.sh G35` → `dist/G35_Assignment2.zip` |

## 1.4 Results at a glance

| Task | Metric | Value |
|---|---|---|
| Naive Bayes (444 test docs) | Accuracy / macro-F1 | 0.962 / 0.960 (0.983 / 0.982 after de-duplication) |
| K-Means (k = 5) | Purity / Rand Index | 0.967 / 0.975 (5-seed mean purity 0.963 ± 0.004) |
| Recommendation | Precision@5 / @10 | 0.933 / 0.920 (relevant = same category) |
| Duplicates | Exact pairs / near-only / dropped | 137 / 26 / 162 |
| Leakage | Test docs with an exact copy in training | 38 |

All figures are in [04_RESULTS.md](04_RESULTS.md) and the report.
