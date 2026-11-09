# IR Assignment 2 — Plan for Full Marks

**Course:** Information Retrieval (AIMLCZG537 / DSECLZG537), S1-26 · **Marks:** 10 · **Deadline:** 9 Nov 2026, 23:59

## 1. Chosen context

**News intelligence for a media-monitoring / business-research firm.** Analysts receive a large stream of
unstructured news articles and need to: auto-tag articles by desk (business, politics, tech…), discover
topical groupings without labels, spot duplicate syndicated stories, and find related coverage for a
story they are researching.

**Dataset:** BBC News raw full-text corpus (Greene & Cunningham, ICML 2006) — real, public, authentic.

| Property | Value |
|---|---|
| Documents | 2,225 `.txt` articles (2004–2005) |
| Raw size | ~5 MB text (9.3 MB on disk) — meets "50–75 docs or ~5 MB" |
| Labels | 5 categories: business, entertainment, politics, sport, tech |
| Metadata | doc_id, title (first line), category, source, period, filename, char/word counts |
| Noise | mixed casing, quotes, punctuation, numbers, £/$ symbols, proper nouns, domain jargon |
| Duplicates | corpus has known exact + near-duplicate articles → used for duplicate analysis |

Text and metadata stored separately: `data/documents.csv` (doc_id, text) and `data/metadata.csv`.

## 2. Rubric → deliverable mapping

| Rubric item (marks) | How we cover it |
|---|---|
| Streamlit workflow & UI (1) | 7-page app: Dashboard, Ingestion, Preprocessing, Mining Hub (NB / K-Means / Evaluation), Recommendation, Performance Analytics, About |
| Preprocessing & features (1) | Configurable pipeline (lowercase, HTML/markup strip, regex tokenization, stop-words, stemming OR lemmatization, n-grams, min-df). **Manual inverted index + TF-IDF + L2 length normalization + cosine**. Before/after vocab & TF plots, Zipf plot, stats |
| Manual NB, K-Means, evaluation (3) | Multinomial NB from scratch (Laplace smoothing, log-probs); K-Means from scratch (k-means++ or random init, assignment, update, stop by tolerance/max-iter, iteration log); Purity & Rand Index from scratch |
| Recommendation (1) | Content-based Top-K via manual cosine on TF-IDF; scores + metadata; explanation + limitations; optional duplicate filtering; precision@K evaluation (same-category hit rate) |
| Visualizations (1) | Actual vs predicted bar chart, confusion matrix heatmap, metrics table, 2D PCA scatter (PCA via numpy SVD, visualization only), cluster-size bar, membership table, comparison chart, rec score bar |
| Results, inferences, report (2) | `report/Report.pdf` generated from real experiment runs (`experiments.py`) answering all 5 compulsory questions with numbers; preprocessing ablation table |
| BITS Virtual Lab demo (1) | **Student action**: run app in Virtual Lab, take screenshots + short screen recording |

## 3. "No built-in library" rule

Prohibited libraries are **not** used for TF-IDF, Naive Bayes, K-Means, Purity, Rand Index, cosine.
scikit-learn is **not installed at all**. Allowed helpers: `numpy` (arrays), `nltk` (stemmer, lemmatizer,
stop-word list only), `pandas`, `plotly`, `streamlit`. PCA (for 2D plot only) is a numpy SVD.

## 4. Project layout

```
assignments/
├── PLAN.md
├── README.md                 # install + run steps
├── requirements.txt
├── app.py                    # Streamlit entry (streamlit run app.py)
├── textmining/
│   ├── preprocessing.py      # configurable pipeline
│   ├── tfidf.py              # inverted index, TF-IDF, L2 norm, cosine (manual)
│   ├── naive_bayes.py        # multinomial NB (manual)
│   ├── kmeans.py             # K-Means w/ step log (manual)
│   ├── metrics.py            # acc/prec/rec/F1, confusion matrix, purity, rand index (manual)
│   ├── recommender.py        # content-based Top-K
│   ├── duplicates.py         # exact (hash) + near-dup (cosine / shingle Jaccard)
│   ├── pca.py                # 2D projection for plotting only
│   └── data_loader.py        # CSV / JSON / zip-of-txt ingestion
├── data/
│   ├── prepare_dataset.py    # raw BBC folder → documents.csv + metadata.csv
│   ├── documents.csv
│   └── metadata.csv
├── experiments.py            # runs all experiments + ablation → report/results.json + figures
├── report/
│   ├── build_report.py       # results → Report.pdf
│   ├── figures/
│   └── Report.pdf
└── G<xx>_Contribution.xlsx   # template — fill names/IDs
```

## 5. Build steps

### Step 1 — Dataset (B)
- Download `bbc-fulltext.zip` from http://mlg.ucd.ie/datasets/bbc.html, unzip to `data/raw_download/bbc/`.
- `prepare_dataset.py`: first line of each file = title, rest = body. Write `documents.csv` (doc_id, text) and
  `metadata.csv` (doc_id, title, category, source, period, filename, n_chars, n_words) — text/metadata separate.
- Ingestion page must also accept user upload: CSV, JSON, ZIP of `.txt` (folder name = label). Auto-assign
  unique doc_id if missing. Show corpus stats right after load.

### Step 2 — Preprocessing & features (C)
- Pipeline toggles: strip markup/URLs, lowercase, regex tokenize, remove numbers/punctuation, stop-words,
  normalizer {none, Porter stem, WordNet lemma}, min token length, n-gram range.
- **Manual inverted index**: `term → {doc_id: tf}`.
- **Manual TF-IDF** (`ltc`): `w = (1+log10 tf) · log10(N/df)`, then divide by `sqrt(Σw²)` (length normalization).
  Prune by min_df / max_df / max_features.
- **Cosine**: dot product of L2-normalized vectors (also show formula `a·b/(|a||b|)`).
- Visuals: before/after table (tokens, vocab size, TTR, hapax), top-20 term bars before vs after, Zipf log-log,
  doc-length histograms, IDF distribution, inverted-index explorer (term → postings/df/idf), similarity heatmap.

### Step 3 — Naive Bayes (D1)
- Multinomial NB from scratch: `log P(c) + Σ tf·log((T_ct+α)/(Σ T_c + α|V|))`, Laplace α=1.
- Manual stratified 80/20 split (seeded). Vocab from **train only** (no leakage).
- Show table doc_id / title / actual / predicted / correct; accuracy, per-class + macro/weighted P/R/F1.
- Extra credit-ish: top indicative words per class, "classify your own text" box.

### Step 4 — K-Means (D2)
- From scratch on TF-IDF rows: init (random or k-means++), assignment (nearest centroid), update (mean),
  stop when no doc changes / centroid shift < tol / max_iter. n_init restarts, keep lowest SSE.
- **Show every step**: initial seed docs, per-iteration log (docs changed, centroid shift, SSE, cluster sizes),
  SSE-vs-iteration chart, stop reason. Top terms per centroid. Optional elbow chart.

### Step 5 — Evaluation (D3)
- **Purity** = `(1/N) Σ_k max_j |ω_k ∩ c_j|` (manual).
- **Rand Index** = `(TP+TN)/C(N,2)`; compute TP/FP/FN/TN from contingency table via `C(n,2)`; verify
  against brute-force O(N²) pair loop on a 200-doc sample (show both match in app/report).
- Comparison table: NB accuracy/F1 vs K-Means purity/RI + text explaining supervised vs unsupervised.

### Step 6 — Visualizations (D4)
Actual-vs-predicted grouped bar · confusion-matrix heatmap · metrics table · 2D PCA scatter colored by
cluster (PCA via numpy eigen-decomposition, plot only) · cluster-size bar · membership table
(doc_id, title, true label, cluster) · contingency heatmap · NB-vs-KM comparison chart.

### Step 7 — Recommendation (E)
- Content-based Top-K: cosine of TF-IDF vectors. Input = pick a doc (search by title) OR free text.
- Output table: rank, score, doc_id, title, category, source, period + top shared terms (why recommended).
- Option "hide near-duplicates (cos ≥ 0.9)" so analysts get new info.
- Evaluate: Precision@5 using same-category as relevance proxy.
- Write explanation + limitations (vocabulary mismatch/synonyms, no semantics, popularity-blind, cold-start n/a,
  dup flooding, same-event bias). Compare with collaborative (needs user–doc interactions; none here).

### Step 8 — Duplicates (B + F2)
- Exact: MD5 of normalized text. Near: cosine ≥ 0.9, cross-check with 5-shingle Jaccard.
- Measure train/test leakage (test docs with exact copy in train). Toggle "remove duplicates" and rerun.

### Step 9 — Performance analytics page + experiments
Timing per stage + ablation runner. Configs to compare (fill report tables with these):

| Run | Change |
|---|---|
| Baseline | lower + stop + stem + unigram |
| A | no stop-word removal |
| B | lemma instead of stem |
| C | no normalizer |
| D | unigram+bigram |
| E | no lowercasing |
| F | baseline + dedup |

Record: vocab size, NB acc / macro-F1, purity, RI, P@5, leakage count.

**Expected baseline (smoke test, seed 42):** vocab 21,238 → 5,000 after pruning; NB acc ≈ 0.96, macro-F1 ≈ 0.96;
K-Means k=5 purity ≈ 0.97, RI ≈ 0.97 (converged ~31 iters); 137 exact-dup groups, 162 docs to drop; P@5 ≈ 0.94.

### Step 10 — Report PDF (F — compulsory, 2 marks)
Sections: business context · dataset description · implementation approach · preprocessing & feature choices ·
classification results (predicted labels, metrics, confusion matrix) · clustering results (assignments,
purity, RI, scatter, size chart) · comparison · recommendation results (Top-K table + **screenshot**) ·
**answers to all 5 questions with numbers from ablation** · limitations & improvements · Virtual Lab evidence.

Key arguments for Q1–Q5:
1. Stop-words/stemming shrink vocab, raise NB & purity; bigrams add phrases ("prime_minister") but sparsity.
2. Duplicates → inflated test accuracy (leakage), tight artificial clusters, recommendation lists flooded with copies.
3. Content-based: needs only text, explains via shared terms, works for new docs; collaborative needs
   interaction logs, finds serendipitous items, suffers cold start. Hybrid best in production.
4. Auto-routing articles to desks, discovering emerging themes, de-duplicating syndicated feeds, faster research.
5. NB (supervised, uses labels) vs K-Means (unsupervised) — different metrics, not directly comparable;
   limitations: bag-of-words, fixed k, 2004-05 data; improvements: embeddings, online learning, active labeling.

### Step 11 — Packaging
- `README.md`: `python -m venv .venv`, `pip install -r requirements.txt`, `streamlit run app.py`.
- `requirements.txt`: streamlit, pandas, numpy, nltk, plotly, openpyxl, matplotlib, reportlab (no scikit-learn).
- `Gxx_Contribution.xlsx` with columns exactly: “Student Registration Number”, “Name”, “Percentage of contribution out of 100%”.

## 5b. Audit fixes (9 Nov 2026 — deadline today)

| # | Gap found | Fix |
|---|---|---|
| 1 | requirements missed report deps | add `matplotlib`, `reportlab` to `requirements.txt` |
| 2 | K-Means doc claimed Euclidean ≡ cosine (false: centroids not unit length) | comment corrected ✅ |
| 3 | Lemmatizer crashes if WordNet download blocked in Virtual Lab | rule-based fallback ✅ |
| 4 | K-Means single seed unstable (dedup run: purity 0.97 → 0.79) | experiments report K-Means over 5 seeds (mean ± std); app shows SSE per restart |
| 5 | Report needs recommendation screenshot | headless-browser local screenshots auto-embedded; replace with Virtual Lab ones |
| 6 | Leakage not quantified | report `test_leakage_docs` (38 test docs have exact copy in train → NB 0.962 with dups vs 0.983 deduped) |
| 7 | TF-IDF fit on full corpus | state in report: used for unsupervised clustering/rec only; NB vocab train-only |

**Build order (priority):** app.py → experiments.py + figures → Report.pdf → README/requirements/xlsx → screenshots → zip.

## 6. Manual tasks (yours)

- [ ] Fill group id, names, registration numbers in contribution sheet.
- [ ] Run app in **BITS Virtual Lab** — screenshots of every page + short screen recording (1 mark).
- [ ] Put screenshots in report (Recommendation screenshot mandatory).
- [ ] Zip as `Gxx_Assignment2.zip` (exclude `.venv`, `data/raw_download`). Submit before 9 Nov 2026 23:59.

## 7. Full-marks checklist

- [ ] No sklearn anywhere; TF-IDF, NB, K-Means, Purity, RI, cosine all hand-written
- [ ] Every step demonstrated in Streamlit UI (not notebook)
- [ ] K-Means steps visible (init, assign, update, stop)
- [ ] All 9 required charts/tables present
- [ ] All 5 inference questions answered with experimental numbers
- [ ] Report has Top-K table + recommendation screenshot
- [ ] README, dataset, code, PDF, contribution xlsx, demo evidence in zip

## 8. Status (9 Nov 2026) — BUILT ✅

App (7 pages, 0 exceptions), experiments, 36-page `report/Report.pdf`, README, requirements, contribution template, local screenshots, `make_zip.sh` all done.
Encoding bug fixed (raw files UTF-8; `£` was showing as `Â£`).

**Finish steps (yours):**
1. Rename `Gxx_Contribution.xlsx` → `G<id>_Contribution.xlsx`; fill reg. numbers, names, %.
2. Replace `Gxx` + member table on the title page in `report/build_report.py`.
3. In BITS Virtual Lab: `pip install -r requirements.txt` → `streamlit run app.py`; screenshot each page into `screenshots/` (keep the name `recommendation.png` for the Recommendation page); record a short screen video (put the `.mp4`/`.mov` in the project root).
4. Create an empty `screenshots/VIRTUAL_LAB.txt` → only then the report states it ran in the Virtual Lab.
5. `python report/build_report.py` → `./make_zip.sh G<id>` → submit `dist/G<id>_Assignment2.zip`.

## 9. Status update (9 Nov 2026, later)

* Report rewritten as 11 concise pages in the assignment's section order (was 36).
* Test suite: **248 tests, all passing** (`./run_tests.sh`). See `docs/05_TESTING.md`.
* Detailed docs in `docs/01..05`. README links to all of them.
* Code on GitHub: https://github.com/shreyanshsharma88/IR-02 (the zip for the portal is still built with `./make_zip.sh G<id>`).
