# Text Mining & Document Intelligence System — News Intelligence

Information Retrieval (AIMLCZG537/DSECLZG537) S1-26 — Assignment 2 · Group **Gxx** · Repository: <https://github.com/shreyanshsharma88/IR-02>

An end-to-end Streamlit application for a **media-monitoring / business-research firm**: it ingests a
collection of news articles, preprocesses them with a configurable pipeline, builds an inverted index and
TF-IDF features, classifies articles into desks (Naive Bayes), discovers groupings (K-Means), evaluates them
(accuracy/P/R/F1, Purity, Rand Index), detects duplicate stories and recommends related articles (Top-K
content-based recommendation).

> **Hand-implemented algorithms.** The inverted index, TF-IDF (ltc + L2 length normalisation), cosine
> similarity, Multinomial Naive Bayes, K-Means, Purity and Rand Index are written from scratch in the
> `textmining/` package. scikit-learn is **not** used anywhere (and is not in `requirements.txt`). numpy is
> only an array container; NLTK only provides the Porter stemmer, WordNet lemmatiser and stop-word list.

## Documentation

| Doc | Contents |
|---|---|
| [docs/01_ASSIGNMENT_AND_SOLUTION.md](docs/01_ASSIGNMENT_AND_SOLUTION.md) | What the assignment asks, the rubric, our choices, and a requirement → code → app page map |
| [docs/02_ARCHITECTURE.md](docs/02_ARCHITECTURE.md) | Architecture, module map, every algorithm with its formula, design decisions |
| [docs/03_USER_GUIDE.md](docs/03_USER_GUIDE.md) | Install, run, test, page walkthrough, Virtual Lab demo script, packaging, troubleshooting |
| [docs/04_RESULTS.md](docs/04_RESULTS.md) | All experimental results and the ablation |
| [docs/05_TESTING.md](docs/05_TESTING.md) | Test strategy, how to run the tests, latest results |
| [report/Report.pdf](report/Report.pdf) | The submitted technical report |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
streamlit run app.py          # the app, at http://localhost:8501
./run_tests.sh                # the full automated test suite
```

## Dataset

BBC News raw full-text corpus (Greene & Cunningham, ICML 2006, <http://mlg.ucd.ie/datasets/bbc.html>):
2,225 real articles from 2004-2005, 5 categories (business, entertainment, politics, sport, tech),
~5.0 MB of raw text, with 137 exact-duplicate pairs and 26 further near-duplicate pairs.
Text and metadata are stored separately:

| File | Columns |
|---|---|
| `data/documents.csv` | `doc_id, text` |
| `data/metadata.csv` | `doc_id, title, category, source, period, filename, n_chars, n_words` |

The prepared CSVs are included. To rebuild them from the raw download
(`bbc-fulltext.zip` unzipped to `data/raw_download/bbc/`):

```bash
python data/prepare_dataset.py
```

The app also accepts uploads: CSV, JSON, or a ZIP of `.txt` files (one folder per category).

## Folder structure

```
.
├── app.py                    # Streamlit entry point (streamlit run app.py)
├── ui/                       # one module per Streamlit page + shared state/styles
├── textmining/               # all algorithms (hand-written)
│   ├── data_loader.py        # CSV / JSON / ZIP ingestion, text vs metadata split
│   ├── preprocessing.py      # configurable pipeline + corpus statistics
│   ├── tfidf.py              # inverted index, TF-IDF (ltc), L2 normalisation, cosine
│   ├── naive_bayes.py        # multinomial NB (Laplace), stratified split
│   ├── kmeans.py             # K-Means with step log (k-means++/random, restarts)
│   ├── metrics.py            # confusion matrix, P/R/F1, Purity, Rand Index
│   ├── recommender.py        # content-based Top-K, explanations, Precision@K
│   ├── duplicates.py         # exact (MD5) + near (cosine / 5-shingle Jaccard) duplicates
│   ├── pca.py                # 2-D projection for plotting only
│   └── pipeline.py           # end-to-end run used by the app and the ablation
├── data/                     # documents.csv, metadata.csv, prepare_dataset.py
├── experiments.py            # reproducible experiments -> report/results.json + figures
├── report/
│   ├── build_report.py       # results.json + figures -> Report.pdf
│   ├── results.json          # every number quoted in the report
│   ├── figures/              # every figure in the report
│   ├── nb_test_predictions.csv, kmeans_membership.csv
│   └── Report.pdf
├── screenshots/              # app screenshots (auto-embedded in the report)
├── docs/                     # detailed documentation (see table above)
├── tests/                    # pytest suite: unit, integration, end-to-end, compliance
├── run_tests.sh              # run the test suite (--fast skips slow tests)
├── make_zip.sh               # build dist/G<id>_Assignment2.zip
├── requirements.txt
└── Gxx_Contribution.xlsx
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional - NLTK data (the app has offline fallbacks for stop-words and lemmatisation)
python -m nltk.downloader stopwords wordnet omw-1.4
```

## Run the application

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

## Reproduce the experiments and the report

```bash
python experiments.py && python report/build_report.py
```

`experiments.py` (seed 42, ~2 minutes) writes `report/results.json`, `report/figures/*.png` and two CSVs;
`build_report.py` turns them into `report/Report.pdf`. Every PNG placed in `screenshots/` is embedded in the
"Evidence of execution in BITS Virtual Lab" section; `screenshots/recommendation.png` is also used as the
recommendation-interface screenshot.

## Running in the BITS Virtual Lab

1. Copy / unzip the project folder into the Virtual Lab machine and open a terminal in it.
2. Create the environment and install dependencies (Setup above). If the lab has no internet access for
   NLTK data, skip the `nltk.downloader` step — built-in fallbacks are used.
3. Start the app: `streamlit run app.py` (add `--server.port 8502` if 8501 is busy) and open the printed URL
   in the lab browser.
4. Walk through every page, take screenshots (save them as PNG in `screenshots/`, e.g.
   `01_dashboard.png`, `05_recommendation.png`, plus `recommendation.png`) and record a short screen video.
5. Re-run `python report/build_report.py` to embed the screenshots in the PDF.

## Page guide

| Page | What it shows |
|---|---|
| Dashboard | Corpus overview: documents, size, category mix, lengths, duplicate statistics |
| Data Ingestion | Load the bundled corpus or upload CSV/JSON/ZIP; text/metadata separation; corpus statistics |
| Preprocessing & Features | Pipeline toggles; before/after vocabulary and term-frequency distributions; Zipf plot; inverted index explorer; TF-IDF and cosine similarity |
| Text Mining Hub | Naive Bayes (actual vs predicted, metrics, confusion matrix), K-Means (initialisation, assignment, update, stopping, 2-D scatter, cluster sizes, membership), Purity / Rand Index and NB vs K-Means comparison |
| Recommendation | Top-K related documents for a selected document or free text, with scores, metadata, shared terms and duplicate hiding |
| Performance Analytics | Stage timings and the preprocessing / de-duplication ablation runner |
| Inferences & Discussion | Experimental evidence and answers to the five compulsory questions |

## Key results (seed 42, full corpus)

| Task | Result |
|---|---|
| Naive Bayes (444 test docs) | accuracy 0.962, macro-F1 0.960 (0.983 after de-duplication) |
| K-Means (k = 5) | purity 0.967, Rand Index 0.975 (5-seed mean purity 0.963) |
| Recommendation | Precision@5 = 0.933, Precision@10 = 0.920 (same-category proxy) |

See `report/Report.pdf` for the full analysis.
