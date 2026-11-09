# 4. Experimental Results

All numbers come from `python experiments.py` (seed 42, stored in `report/results.json`). They are the same
numbers the app shows with default settings.

## 4.1 Corpus

| Item | Value |
|---|---|
| Documents / categories | 2,225 / 5 (business 510, entertainment 386, politics 417, sport 511, tech 401) |
| Raw text | 5.03 M characters, 961,301 raw tokens, vocabulary 37,981 |
| Exact duplicates | 137 pairs (274 documents involved) |
| Near-only duplicates (cos ≥ 0.9) | 26 pairs (mean 5-shingle Jaccard 0.887, min 0.641) |
| Duplicate pairs across categories | 0 |
| Dropped by de-duplication | 162 → 2,063 documents left |

## 4.2 Preprocessing (baseline: lowercase + stop-words + Porter stem + unigrams)

| Stage | Tokens | Vocabulary |
|---|---|---|
| Raw tokens | 961,301 | 37,981 |
| Lowercase, punctuation removed | 843,375 | 33,096 |
| + stop-words, numbers, stemming (baseline) | 451,770 (−53%) | 21,238 (−44%) |
| After TF-IDF pruning (min_df 2, max_df 0.9, top 5,000) | — | 5,000 |

Stop-word removal removes most of the **tokens**. Stemming removes most of the **vocabulary**.

## 4.3 Naive Bayes (1,781 train / 444 test, α = 1)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business | 0.980 | 0.951 | 0.965 | 102 |
| entertainment | 0.973 | 0.935 | 0.954 | 77 |
| politics | 0.941 | 0.952 | 0.946 | 83 |
| sport | 0.990 | 0.980 | 0.985 | 102 |
| tech | 0.919 | 0.988 | 0.952 | 80 |
| **macro** | **0.960** | **0.961** | **0.960** | 444 |
| **accuracy** | | | **0.962** (427 correct / 17 wrong) | |

**Leakage:** 38 test articles have an exact copy in the training set, and the model classifies all of them
correctly. After de-duplication there is no leakage and accuracy is **0.983**.

## 4.4 K-Means (k = 5, k-means++, 5 restarts)

| Item | Value |
|---|---|
| Stop | Converged at iteration 31: no document changed cluster |
| SSE | 2,072.77 |
| **Purity** | **0.967** |
| **Rand Index** | **0.975** (brute-force check on a 300-document sample matches exactly) |
| Seed stability, 5 restarts (5 seeds) | Purity 0.963 ± 0.004, RI 0.972 ± 0.003 |
| Seed stability, 1 restart (5 seeds) | Purity 0.922 ± 0.070 (worst seed 0.782) |
| De-duplicated corpus, 5 restarts (5 seeds) | Purity 0.927 ± 0.068. Seed 42 alone gets 0.792 (a poor local optimum, higher SSE) |

## 4.5 Naive Bayes vs K-Means

| Model | Type | Metric | Value |
|---|---|---|---|
| Naive Bayes | supervised | Accuracy / macro-F1 | 0.962 / 0.960 |
| K-Means | unsupervised | Purity / Rand Index | 0.967 / 0.975 |

The metrics measure different things. NB is scored against labels it was trained on. K-Means never sees the
labels, so cluster–class agreement is measured afterwards. The BBC desks happen to be lexically very distinct,
which is why K-Means almost recovers them. K-Means is much less **stable** than NB, though: it depends on the
random start.

## 4.6 Recommendation (content-based, cosine on TF-IDF)

| Setting | P@5 | P@10 |
|---|---|---|
| Full corpus | 0.933 | 0.920 |
| Near-duplicates hidden | 0.931 | 0.919 |
| De-duplicated corpus | 0.928 | 0.915 |

Per-category P@5: sport 0.986 · tech 0.935 · politics 0.930 · business 0.904 · entertainment 0.902.
323 documents (14.5%) have a near-duplicate in their own Top-5 unless duplicates are hidden. For example,
query BUS-007's #1 result is its exact copy BUS-253 with score 1.0.

## 4.7 Preprocessing ablation

| Configuration | Vocabulary | NB acc | NB macro-F1 | K-Means purity (5-seed mean) | P@5 |
|---|---|---|---|---|---|
| **Baseline** (lower+stop+stem) | 21,238 | 0.962 | 0.960 | 0.959 | 0.933 |
| No stop-word removal | 21,332 | 0.960 | 0.958 | 0.911 | 0.936 |
| Lemmatisation instead of stemming | 27,572 | 0.966 | 0.965 | 0.915 | 0.938 |
| No normaliser | 30,998 | 0.966 | 0.965 | 0.889 | 0.936 |
| Unigrams + bigrams | 321,720 | 0.964 | 0.962 | 0.859 | 0.946 |
| No lowercasing, no stemming | 35,047 | 0.969 | 0.967 | 0.889 | 0.949 |
| Baseline + de-duplication | 21,227 | **0.983** | **0.982** | 0.872 | 0.917 |

What the ablation shows:
* NB barely changes across preprocessing choices (0.960–0.969). That range is smaller than the variation from
  using different train/test splits (0.970 ± 0.009).
* Clustering is the most sensitive step. Stemming plus stop-word removal gives the most stable clusters.
  Bigrams blow up the vocabulary 15× and make clustering less stable.
* Recommendation P@5 goes up slightly with richer vocabularies (bigrams, case kept), because more specific
  terms are kept.
* De-duplication matters more than any preprocessing choice: it removes the leakage that was inflating NB's
  apparent result and stops copies filling the recommendation lists.
* "No lowercasing" alone gives exactly the baseline numbers, because the Porter stemmer lowercases its output.
  That is why the "no lowercasing, no stemming" row exists.
