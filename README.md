# Text Mining & Document Intelligence System

Information Retrieval (AIMLCZG537/DSECLZG537) S1-26, Assignment 2, Group **G35**
Repository: <https://github.com/shreyanshsharma88/IR-02>

A Streamlit app for analysing news articles (BBC News, 2,225 articles, 5 categories). It covers preprocessing,
an inverted index with TF-IDF, cosine similarity, Naive Bayes classification, K-Means clustering, Purity and
Rand Index, duplicate detection and Top-K recommendation. All of these algorithms are written from scratch in
`textmining/`; scikit-learn is not used.

## Team members

| # | Name | Registration number | Email |
|---|---|---|---|
| 1 | LAKSHYA KANCHAN | 2025AG05062 | 2025ag05062@wilp.bits-pilani.ac.in |
| 2 | SHREYANSH SHARMA | 2025AG05772 | 2025ag05772@wilp.bits-pilani.ac.in |
| 3 | VINAY PRATAP SINGH BHADAURIA | 2025AG05689 | 2025ag05689@wilp.bits-pilani.ac.in |
| 4 | NAIR ABHIJITH VIJAYAKUMAR | 2025AG05914 | 2025ag05914@wilp.bits-pilani.ac.in |
| 5 | ABHISHEK GHOSH | 2025AG05076 | 2025ag05076@wilp.bits-pilani.ac.in |

## BEFORE SUBMISSION (DEADLINE: 9 NOV 2026, 23:59)

1. **CHECK `G35_Contribution.xlsx` (REGISTRATION NUMBERS, NAMES AND PERCENTAGES - CURRENTLY 100% FOR EVERY MEMBER). GROUP ID AND MEMBERS ARE SET AT THE TOP OF `report/build_report.py`.**
2. **RUN THE APP IN THE BITS VIRTUAL LAB AND SAVE SCREENSHOTS INTO `screenshots/`, KEEPING THE NAME `recommendation.png` FOR THAT PAGE. RECORD A SHORT VIDEO, THEN CREATE AN EMPTY `screenshots/VIRTUAL_LAB.txt`; ONLY THEN DOES THE REPORT SAY IT RAN IN THE LAB.**
3. **RUN `python report/build_report.py`, THEN `./make_zip.sh G35`, AND UPLOAD `dist/G35_Assignment2.zip` TO THE COURSE PORTAL.**
4. **OPTIONALLY COMMIT AND PUSH THE FINAL VERSION.**

## Setup and run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

NLTK data (`python -m nltk.downloader stopwords wordnet omw-1.4`) is optional; built-in fallbacks are used when
it is missing.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                    # full suite
pytest -m "not slow"      # skip the slow UI tests
```

## Report

```bash
python experiments.py             # writes report/results.json and report/figures/
python report/build_report.py     # writes report/Report.pdf
```

## Layout

```
app.py              Streamlit entry point
ui/                 one module per page, plus the cached pipeline (state.py)
textmining/         preprocessing, TF-IDF, Naive Bayes, K-Means, metrics, duplicates, recommender
data/               documents.csv (doc_id, text), metadata.csv, prepare_dataset.py
experiments.py      experiments behind the report
report/             build_report.py, Report.pdf, results and figures
tests/              pytest suite
make_zip.sh         builds the submission zip
```

Dataset: D. Greene and P. Cunningham, ICML 2006, <http://mlg.ucd.ie/datasets/bbc.html>.
`python data/prepare_dataset.py` rebuilds the CSVs from the raw download in `data/raw_download/bbc/`.
