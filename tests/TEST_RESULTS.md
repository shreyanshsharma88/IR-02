# Test Results — Text Mining & Document Intelligence System

**Date:** 2026-11-09
**Command:** `./run_tests.sh` (equivalent: `.venv/bin/python -m pytest -v -p no:cacheprovider`)
**Fast mode:** `./run_tests.sh --fast` skips the 4 tests marked `slow` (244 tests, about 45 s)

## Environment

| Item | Version |
|---|---|
| OS | macOS (Darwin 27.0.0) |
| Python | 3.14.7 (`.venv`) |
| pytest | 9.2.0 |
| streamlit | 1.66.1 |
| pandas / numpy / nltk | 3.1.1 / 2.5.4 / 3.10.4 |
| scikit-learn | **not installed** (the assignment forbids it; `test_compliance` checks this) |

## Summary

**248 passed, 0 failed, 0 skipped, in about 89 s.** The full run includes 4 slow tests: two ablation buttons, the elbow curve and the live server. The pytest config and runner are in `pytest.ini` (testpaths, `slow`/`e2e` markers), `run_tests.sh` and `requirements-dev.txt`. `conftest.py` loads the corpus once per session and builds the baseline tokens, TF-IDF and `run_experiment` results (with and without dedup) once, so the suite stays fast.

| File | Type | Tests | Result | Time (s) | What it verifies |
|---|---|---:|---|---:|---|
| `test_preprocessing.py` | unit | 29 | 29 pass | <1 | Each `PreprocessConfig` toggle (lowercase, markup/URL/entity strip, numbers, punctuation, stop-words, custom stop-words, none/stem/lemma, min length, n-grams with `a_b` bigrams), the WordNet-missing lemma fallback and the stemmer fallback (monkeypatched), exact `corpus_stats` values, and the real-corpus baseline vocabulary of 21,238 |
| `test_tfidf.py` | unit | 21 | 21 pass | <1 | A hand-computed corpus: postings and df, idf = log10(N/df), smooth idf, `ltc` weight (1+log10 tf)·idf, raw tf, L2 row norm = 1, and min_df/max_df/max_features pruning. Also: `transform()` of a training doc equals its matrix row, unknown terms are ignored, matrix cosine equals the pure-Python `cosine_pair_scalar`, identical docs score 1 and disjoint docs 0 |
| `test_naive_bayes.py` | unit | 13 | 13 pass | <1 | IIR Example 13.1: P(Chinese\|c)=3/7, P(Tokyo\|c)=P(Japan\|c)=1/14, P(·\|c̄)=2/9, priors 3/4 and 1/4, scores ≈0.0003 vs 0.0001, prediction = yes. Also: `predict_proba` sums to 1, unknown terms are ignored, alpha works, `stratified_split` has the right proportions, is disjoint and is seed-deterministic, and the vocabulary comes from training docs only |
| `test_kmeans.py` | unit | 15 | 15 pass | 1.5 | Synthetic blobs are recovered with purity = RI = 1 (random and k-means++ init). SSE never increases. `stop_reason` covers no-change, tol and max_iter. Same seed gives the same result. `init_indices` has length k, empty clusters are re-seeded, and `n_init` keeps the run with the lowest SSE. Elbow SSE decreases. Real-data run: 31 iterations, SSE ≈ 2072.77 |
| `test_metrics.py` | unit | 15 | 15 pass | <1 | IIR Fig. 16.4: purity 12/17 ≈ 0.71, TP/FP/FN/TN = 20/20/24/72, RI = 0.68. The fast RI equals brute force on 8 random labelings. Hand-checked classification report: accuracy, per-class P/R/F1, macro, weighted, confusion matrix with rows = true labels, and zero-division handling |
| `test_duplicates.py` | unit + real data | 10 | 10 pass | 2.8 | Exact duplicates are found regardless of case or whitespace. Near-duplicate pairs are found above the threshold, and blocking does not change the result. Shingles and Jaccard are correct. `drop_indices` keeps the first doc of each group. Real corpus: 137 exact groups (all pairs), 163 pairs with cos ≥ 0.9, 162 docs dropped, 2,063 kept |
| `test_recommender.py` | unit + real data | 21 | 21 pass | 0.5 | The query doc is never returned, results are sorted descending, exactly k results come back, and scores are in [0,1] and equal the cosine. `dup_threshold` hides the exact copy (BUS-007 → BUS-253). Free-text queries work, and empty or unknown queries return `[]`. Explanations are correct, k > corpus size is handled, and real-data P@5 is > 0.85 |
| `test_data_loader.py` | unit | 22 | 22 pass | <1 | `load_default` gives shapes 2225×2, unique ids, metadata without a `text` column, no mojibake (no `Â` or `â€`, `£` present) and the class counts. Uploads: CSV with and without id/label (duplicate ids made unique), JSON list, `{"documents": …}`, columnar JSON, JSONL (multi-line and single-line), ZIP with a folder per category, flat ZIP and ZIP without .txt files. Unsupported formats raise an error |
| `test_pipeline.py` | integration | 9 | 9 pass | 3.5 | Full-corpus `run_experiment` baseline: NB 0.9617, purity 0.9667, RI 0.9749, P@5 > 0.85, leakage = 38. The dedup run gives 2,063 docs, 162 dropped and leakage 0. The corpus is ≈5 MB. Runs are deterministic. A copy of `data/prepare_dataset.py` run in a temp dir regenerates byte-identical CSVs, and sampled CSV rows match the raw files |
| `test_app_e2e.py` | e2e | 30 | 30 pass | 78 | AppTest renders all 7 pages with no exceptions or `st.error`. It also drives the NB and K-Means forms, the classify-your-text box, the `km_color` radio, the elbow (`km_elbow_btn`), `run_seeds`, `run_ablation`, `inf_run_ablation`, `w_dedup` on then a visit to every page (leakage 0, 2,063 docs), `w_dup_threshold`, and the preprocessing form (lemma + custom stop-words). Recommendation: `rec_doc`, `rec_k` (5/10/3 rows), `rec_hide`, `rec_thr` and free-text `rec_mode`, including the no-match warning. Uploads are injected as CSV, unlabelled ZIP and JSON, every page is visited, and "Load default" works again. A live server (`streamlit run --server.headless true --server.port 8611`) returns `/_stcore/health` = `ok` |
| `test_compliance.py` | static | 53 | 53 pass | <1 | An AST scan of every project `.py` file (excluding .venv, dist and dot-dirs) finds no imports of sklearn, gensim, scipy.cluster/spatial/sparse, nltk.classify/cluster, torch and similar libraries. sklearn is not in requirements or installed. The deliverables exist (app.py, README.md, requirements.txt, report/Report.pdf with a valid `%PDF` header, data CSVs, prepare_dataset.py). `*_Contribution.xlsx` has the exact 3 headers |
| `test_experiments.py` | output check | 10 | 10 pass | <1 | `report/results.json` has the expected top-level keys and sensible values (2,225 docs, 137 exact groups, a 2225×5000 matrix, metrics in [0,1], baseline leakage 38, dedup rows with leakage 0, sorted recommendation scores). Every listed figure exists and is a PNG, and the prediction and membership CSVs exist. This file does not run `experiments.py`, and it skips if the outputs are missing |
| **Total** | | **248** | **248 pass** | **≈89** | |

## Bugs found and fixed

1. **`textmining/data_loader.py` crashed on a single-record JSON or one-line JSONL upload.** `json.loads` returned a dict, and `pd.DataFrame(dict_of_scalars)` raised *"If using all scalar values, you must pass an index"*. The fix: a dict that has no `documents` key and no list values is now wrapped as `[obj]`, so it becomes a one-row frame. Test: `test_data_loader.py::test_single_line_jsonl_upload`.
   *This fix does not change any experiment numbers.* It only affects uploads, and the default BBC pipeline, `experiments.py` and `results.json` are not touched.

No other defects were found. Two observations that are not bugs:
- With the Porter stemmer, `lowercase=False` still produces lowercase tokens, because NLTK's stemmer lowercases its output. This is already documented in the report and on the Inferences page. `test_porter_stemmer_lowercases_even_when_lowercase_off` pins this behaviour.
- `classification_report([], [])` would divide by zero in the weighted average. This cannot happen in the app, because `stratified_split` always puts at least one document in the test set, so it was left as is.

## Notes

- AppTest cannot drive `st.file_uploader`. The upload tests therefore inject the dataset through `ui.state.load_upload`, which is the same function the "Ingest uploaded file" button calls, and then exercise every page.
- The live-server test falls back to a free port if 8611 is busy.
- The prepare-dataset tests skip when `data/raw_download/bbc` is absent, as it is in the submission zip.
