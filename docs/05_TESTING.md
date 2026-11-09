# 5. Testing

## 5.1 How to run

```bash
pip install -r requirements-dev.txt     # pytest
./run_tests.sh                          # full suite: 248 tests, about 80 s
./run_tests.sh --fast                   # skip the 4 "slow" tests (ablation / elbow / live server): about 45 s
./run_tests.sh -k tfidf                 # any pytest arguments are passed through
.venv/bin/python -m pytest tests/test_metrics.py -v
```

Configuration lives in `pytest.ini`: `testpaths = tests`, markers `slow` and `e2e`. Shared fixtures are in
`tests/conftest.py`: the dataset and a baseline pipeline are built once per session, so the suite stays fast.

## 5.2 Strategy

| Level | What it proves | Files |
|---|---|---|
| **Unit, checked against textbooks** | The hand-written algorithms give the published answers | `test_naive_bayes` (Manning IIR Example 13.1), `test_metrics` (IIR Fig. 16.4), `test_tfidf` (hand-computed corpus) |
| **Unit, behaviour** | Each preprocessing switch, pruning rule, K-Means step, duplicate rule and recommender option does what it says | `test_preprocessing`, `test_kmeans`, `test_duplicates`, `test_recommender`, `test_data_loader` |
| **Integration** | The full pipeline on all 2,225 documents reproduces the reported numbers | `test_pipeline`, `test_experiments` |
| **End-to-end (UI)** | Every Streamlit page and interactive control works, through `streamlit.testing.v1.AppTest` and a real server | `test_app_e2e` |
| **Compliance** | Assignment rules: no forbidden libraries, every deliverable present, correct contribution-sheet headers | `test_compliance` |

## 5.3 What is verified

| File | Tests | Key checks |
|---|---:|---|
| `test_preprocessing.py` | 29 | Every switch (case, markup/URLs, numbers, punctuation, stop-words, stem/lemma/none, min length, n-grams `a_b`, custom stop-words); `corpus_stats`; lemmatiser fallback when WordNet is missing; Porter lowercasing pinned |
| `test_tfidf.py` | 21 | Postings and df exact; `idf = log10(N/df)`; ltc weight `1+log10 tf`; every row has L2 norm 1; min_df / max_df / max_features pruning; `transform` of a training doc equals its row; unknown terms ignored; matrix cosine equals pure-Python cosine; identical → 1, disjoint → 0 |
| `test_naive_bayes.py` | 13 | IIR 13.1: P(Chinese\|c)=3/7, P(Tokyo\|c)=P(Japan\|c)=1/14, P(·\|c̄)=2/9, priors 3/4 and 1/4, scores 0.0003 vs 0.0001, predicts *yes*; posteriors sum to 1; stratified split keeps proportions, is disjoint and repeatable for a seed; vocabulary comes from training data only |
| `test_kmeans.py` | 15 | Separated blobs recovered with purity 1; SSE never increases; stop reason recorded; repeatable for a seed; random and k-means++ both work; k init indices; empty-cluster handling; best of `n_init` runs kept; real data converges |
| `test_metrics.py` | 15 | IIR 16.4: purity 12/17 ≈ 0.71, TP/FP/FN/TN = 20/20/24/72, RI = 0.68; fast RI equals brute force on random labelings; hand-checked P/R/F1, macro, weighted, confusion matrix orientation (rows = true) |
| `test_duplicates.py` | 10 | Exact duplicates ignore case and whitespace; near-duplicates found above the threshold; shingle Jaccard; first copy kept; real data: 137 groups, 163 pairs at cos ≥ 0.9, 162 dropped |
| `test_recommender.py` | 21 | Query itself excluded; results sorted; exactly k returned; scores in [0, 1]; duplicate hiding (BUS-007 → BUS-253); free-text query; empty or unknown query returns []; real-data P@5 > 0.85 |
| `test_data_loader.py` | 22 | Default shapes; unique IDs; text and metadata separate; no mis-encoded characters (no `Â`, `£` present); CSV with and without id/label columns; duplicate IDs made unique; JSON, JSONL, single record; ZIP of category folders; unsupported formats rejected |
| `test_pipeline.py` | 9 | Baseline NB acc 0.9617 > 0.93, purity 0.9667, RI 0.9749, leakage 38; de-duplicated run: 2,063 documents, 0 leakage; repeatable; corpus ≥ 2,225 documents, about 5 MB; `prepare_dataset.py` regenerates byte-identical CSVs |
| `test_app_e2e.py` | 30 | All 7 pages render with no exception and no `st.error`; NB / K-Means forms; classify-your-text; scatter colour switch; elbow; seed stability; ablation (two buttons); dedup switch then all pages; preprocessing form (lemma + custom stop-words); recommendation document, K, hide and threshold controls; free-text mode with the no-match warning; CSV, JSON and ZIP uploads, then back to the default; **live server** `/_stcore/health == ok` |
| `test_compliance.py` | 53 | AST scan of every project `.py` file: no `sklearn`, `gensim`, `scipy.cluster` and similar; every deliverable file present; `Report.pdf` valid; contribution sheet headers exactly "Student Registration Number", "Name", "Percentage of contribution out of 100%" |
| `test_experiments.py` | 10 | `results.json` keys and values make sense; every referenced figure exists and is a PNG |
| **Total** | **248** | |

## 5.4 Latest run

```
Python 3.14.7 | streamlit 1.66.1 | pytest 9.2.0 | macOS
./run_tests.sh   →   248 passed in 80.97s   (exit code 0, PASS)
```

The full per-file log is in [`tests/TEST_RESULTS.md`](../tests/TEST_RESULTS.md).

## 5.5 Bugs found by the tests

| Bug | Fix | Affects results? |
|---|---|---|
| `load_uploaded` crashed on a one-line JSONL file or a single JSON object ("If using all scalar values, you must pass an index") | A single record is wrapped into a one-row table (`textmining/data_loader.py`) | No, uploads only |
| *(earlier, during the build)* Raw files read as latin-1, so `£` showed as `Â£` | Read as UTF-8, with latin-1 as a fallback (`data/prepare_dataset.py`) | No; all metrics were identical after regenerating |

Known, deliberate behaviour (pinned by tests, not bugs):
* With stemming on, turning lowercasing off has no effect, because the Porter stemmer lowercases its output. This is documented in the report.
* `classification_report([], [])` divides by zero. The app can never produce an empty test set (the stratified split keeps at least 1 document per class).

## 5.6 Manual verification

* Every page and tab was screenshotted with a headless browser (Playwright + Chrome), and the images were checked by eye (`screenshots/`).
* Every page of `report/Report.pdf` (11 pages) was rendered and checked for layout and readability.
* The packaged zip was unzipped and every app page run from it, with zero exceptions.
