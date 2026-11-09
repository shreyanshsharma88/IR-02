# 3. User Guide: Install, Run, Test, Demo

## 3.1 Requirements

* Python **3.10 or newer** (developed on 3.14; works on the BITS Virtual Lab's Python 3).
* About 300 MB disk for the virtual environment. No GPU. No internet needed after `pip install`.

## 3.2 Install

```bash
cd <project-folder>
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # streamlit, pandas, numpy, nltk, plotly, matplotlib, reportlab, openpyxl
pip install -r requirements-dev.txt  # pytest (only needed to run the tests)

# optional: NLTK data. If this fails (offline lab), built-in fallbacks are used automatically.
python -m nltk.downloader stopwords wordnet omw-1.4
```

## 3.3 Run the app

```bash
streamlit run app.py
# busy port?   streamlit run app.py --server.port 8502
# no browser?  streamlit run app.py --server.headless true
```

Open the URL it prints (default <http://localhost:8501>). The first load preprocesses and indexes the
2,225 documents in about 5 s. After that, each page renders in about a second.

## 3.4 Run the tests

```bash
./run_tests.sh            # full suite (unit + integration + end-to-end app + live server)
./run_tests.sh --fast     # skip the slow tests (ablation / seed runs in the UI)
pytest tests/test_metrics.py -v   # a single file
```

What each test file checks, and the latest results, are in [`tests/TEST_RESULTS.md`](../tests/TEST_RESULTS.md)
and [05_TESTING.md](05_TESTING.md).

## 3.5 Reproduce the report

```bash
python experiments.py            # about 2 min, seed 42 → report/results.json + report/figures/*.png
python report/build_report.py    # → report/Report.pdf
```

Before submitting, set the group ID and member names in the `GROUP_ID` / `MEMBERS` constants at the top of
`report/build_report.py`.

## 3.6 Page-by-page walkthrough (and what to try)

| # | Page | What to look at | Try this |
|---|---|---|---|
| 1 | **Dashboard** | KPIs (docs, categories, MB, vocabulary, duplicates), category bar chart, length histogram, pipeline status | The sidebar shows the active dataset and configuration |
| 2 | **Data Ingestion** | Documents table and metadata table (kept separate), corpus statistics, unique-ID check, noise examples, exact-duplicate preview | Upload a CSV with a `text` column (plus `category` if you have it), a JSON file, or a ZIP of `category/*.txt` |
| 3 | **Preprocessing & Features** | Tabs: *Before vs after* (stats, top-20 terms, Zipf, lengths), *Live demo*, *TF-IDF & inverted index* (formula, pruning, term lookup), *Cosine similarity*, *Duplicates* | Turn stop-words off and watch the vocabulary and top terms change. Look up the term `econom`. Tick **Remove duplicates before modelling** |
| 4 | **Text Mining Hub → Naive Bayes** | Formula, metrics, predictions table (filter to misclassified), actual-vs-predicted chart, confusion matrix, per-class metrics, top terms, "classify your own text" | Paste a headline such as *"Shares in the bank fell after profit warning"* |
| 4 | **→ K-Means** | Seed documents, iteration log, SSE / documents-changed charts, stop reason, restarts, membership table, PCA scatter, cluster sizes, top terms per cluster, elbow | Change k to 4 or 6, switch init to `random`, colour the scatter by true category |
| 4 | **→ Evaluation & comparison** | Purity, Rand Index with TP/FP/FN/TN, brute-force check, contingency heatmap, NB vs K-Means table and chart, written interpretation | — |
| 4 | **→ Corpus analytics** | Top TF-IDF terms per category, Boolean AND search over the inverted index | Search `oil price` |
| 5 | **Recommendation** | Query document card, Top-K table (score, metadata, shared terms), score chart, P@K | Pick `BUS-007`, then untick **Hide near-duplicates** to see its exact copy appear at score 1.0. Switch to *Free text* |
| 6 | **Performance Analytics** | Stage timings, **Run ablation** (7 configurations), **Seed stability**, CSV download | Click both buttons (about 1 min) |
| 7 | **Inferences & Discussion** | The five compulsory questions answered with the live numbers | Run the ablation first so all numbers fill in |

## 3.7 BITS Virtual Lab demo (1 mark, student action)

1. Copy the zip into the lab VM, unzip it, open a terminal in the folder.
2. Follow 3.2 and 3.3. Open the URL in the lab browser.
3. Screenshot each page into `screenshots/`. Keep the name **`recommendation.png`** for the Recommendation
   page, because the report embeds it in §8. Make sure the lab desktop or URL bar is visible, as proof it ran in the lab.
4. Record a 2–4 minute screen video following the script below. Save it as `demo.mp4` (or `.mov`) in the project root.
5. Create an empty file **`screenshots/VIRTUAL_LAB.txt`**. Only then does the report say it was run in the Virtual Lab.
6. `python report/build_report.py` and then `./make_zip.sh G<id>`.

### Suggested demo script (about 3 min)
1. Dashboard: "2,225 BBC articles, 5 desks, 5 MB, 137 duplicate pairs."
2. Ingestion: show the documents and metadata tables side by side, then the statistics.
3. Preprocessing: toggle stop-words and show the vocabulary change. Look up a term in the inverted index. Show the cosine between two documents.
4. Naive Bayes: metrics, confusion matrix, classify a typed headline.
5. K-Means: iteration log and stop reason, scatter, cluster sizes.
6. Evaluation: Purity and Rand Index next to NB accuracy and F1, then the interpretation text.
7. Recommendation: Top-5 for a document, with the duplicate-hiding toggle on and off.
8. Performance: run the ablation and show the table.

## 3.8 Packaging for submission

```bash
mv Gxx_Contribution.xlsx G07_Contribution.xlsx     # use your group id; fill in reg no / name / %
./make_zip.sh G07                                   # → dist/G07_Assignment2.zip
```

The zip contains the code, `data/` (CSVs plus the original raw zip), the report and figures, screenshots, the
README, docs, tests and the contribution sheet. It leaves out `.venv`, caches and `dist/`.

## 3.9 Troubleshooting

| Symptom | Fix |
|---|---|
| `LookupError: stopwords / wordnet` | Fallbacks should handle it. Otherwise run `python -m nltk.downloader stopwords wordnet omw-1.4` |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| `ModuleNotFoundError: textmining` | Run commands from the project root (the folder containing `app.py`) |
| Upload rejected | It needs a text column named one of `text, content, body, document, review, abstract`. ZIPs need `.txt` files |
| App slow on the first click | Normal: the first run builds the cache (about 5 s). Ablation and seed buttons take about 1 min |
