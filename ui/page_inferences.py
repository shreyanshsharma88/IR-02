"""Inferences & discussion page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .common import header, section
from .page_mining import interpretation
from .page_performance import _run_ablation
from .state import Pipeline

QUESTIONS = [
    "How did the choice of preprocessing steps affect the vocabulary, classification results, clustering results, "
    "and document recommendations?",
    "How can duplicate or near-duplicate documents affect classification, clustering, and recommendation results? "
    "Describe how you detected or handled them.",
    "Explain the recommendation approach selected for your application. Compare content-based and collaborative "
    "recommendation, stating when each approach is preferable and what data each requires.",
    "How do text mining, document classification, clustering, and document recommendation together help domain "
    "experts make better decisions in the selected business context?",
    "Compare the Naive Bayes classification results with the K-Means clustering results using the appropriate "
    "evaluation metrics. Summarize your key experimental findings, limitations, and suggested improvements for "
    "real-world deployment.",
]


def _row(ab: pd.DataFrame, name: str):
    r = ab[ab["run"] == name]
    return None if r.empty else r.iloc[0]


def _q1(p: Pipeline, ab):
    rs, ps = p.raw_stats, p.proc_stats_all
    red = (1 - ps["vocabulary_size"] / max(rs["vocabulary_size"], 1)) * 100
    txt = (f"With the active pipeline (`{p.cfg.label()}`) the raw vocabulary of **{rs['vocabulary_size']:,}** token "
           f"types falls to **{ps['vocabulary_size']:,}** ({red:.1f}% smaller). Average document length falls from "
           f"{rs['avg_tokens_per_doc']:.0f} to {ps['avg_tokens_per_doc']:.0f} tokens; after min_df/max_df pruning "
           f"**{len(p.tfidf.vocab):,}** TF-IDF features remain. Lowercasing and punctuation/number removal merge "
           "surface variants; stop-word removal deletes high-frequency, low-information words that otherwise dominate "
           "cosine similarity; stemming groups word forms (econom-y/-ic/-ies).")
    if ab is None:
        return txt + ("\n\n*Run the ablation on the Performance Analytics page (or with the button below) to add the "
                      "per-configuration comparison.*")
    b = _row(ab, "Baseline")
    parts = [txt, "\n\nAblation results (NB accuracy / macro-F1 · K-Means purity / RI · P@5 · vocabulary):\n"]
    for _, r in ab.iterrows():
        parts.append(f"- **{r['run']}**: NB {r['nb_accuracy']:.3f} / {r['nb_macro_f1']:.3f} · KM "
                     f"{r['km_purity']:.3f} / {r['km_rand_index']:.3f} · P@5 {r['precision_at_5']:.3f} · "
                     f"vocab {int(r['vocab_raw']):,} → {int(r['vocab_tfidf']):,}")
    pp = ab[ab["run"] != "Baseline+dedup"]
    if b is not None and len(pp) > 1:
        best_nb = pp.loc[pp["nb_accuracy"].idxmax()]
        best_km = pp.loc[pp["km_purity"].idxmax()]
        worst_km = pp.loc[pp["km_purity"].idxmin()]
        nb_spread = pp["nb_accuracy"].max() - pp["nb_accuracy"].min()
        km_spread = pp["km_purity"].max() - pp["km_purity"].min()
        more = "more" if km_spread > nb_spread else "less"
        parts.append(
            f"\n\nAcross the preprocessing variants Naive Bayes accuracy ranges from {pp['nb_accuracy'].min():.3f} to "
            f"{pp['nb_accuracy'].max():.3f} (best: *{best_nb['run']}*); class-conditional likelihoods down-weight "
            f"uninformative words automatically. K-Means purity spans {worst_km['km_purity']:.3f} "
            f"(*{worst_km['run']}*) to {best_km['km_purity']:.3f} (*{best_km['run']}*). Clustering is {more} "
            f"sensitive to preprocessing (spread {km_spread:.3f} vs {nb_spread:.3f}) because unnormalised variants and "
            "frequent words affect distances to centroids. "
            f"Recommendation P@5 ranges from {pp['precision_at_5'].min():.3f} to {pp['precision_at_5'].max():.3f}; bigrams add "
            "phrase features (e.g. `prime_minist`) at the cost of a larger, sparser vocabulary. Where two "
            "configurations give identical numbers (e.g. lowercasing on/off with Porter stemming, which lowercases "
            "anyway) the step is redundant given the others.")
    return "\n".join(parts)


def _q2(p: Pipeline, ab):
    rep = p.dups
    exact = rep["exact_groups"]
    pairs = rep["pairs"]
    n_near = sum(1 for r in pairs if r["type"] == "near")
    txt = (f"**Detection.** Exact duplicates: MD5 hash of whitespace/case-normalised text → **{len(exact)} groups** "
           f"({sum(len(g) for g in exact)} documents). Near duplicates: cosine of TF-IDF vectors ≥ "
           f"{p.dup_threshold:.2f}, cross-checked with 5-word-shingle Jaccard → {len(pairs):,} pairs ({n_near} "
           f"non-identical). A union-find over all pairs keeps one representative per group, so **"
           f"{len(rep['drop_indices'])} documents** would be dropped when 'Remove duplicates before modelling' is on "
           f"(currently **{'ON' if p.dedup else 'OFF'}**).\n\n"
           f"**Classification.** Copies split across train and test leak information: in the current split "
           f"**{p.nb['leakage']} test documents** have an identical copy in training, so the test article is "
           "effectively already seen and accuracy is optimistic.\n\n"
           "**Clustering.** Duplicates form artificially tight groups, add extra weight to their topic in centroid "
           "means, and inflate TP pair counts in the Rand index.\n\n"
           "**Recommendation.** A syndicated copy scores cosine ≈ 1.0 and takes up a Top-K slot without adding information; "
           "the recommender therefore offers a 'hide near-duplicates' filter (cos ≥ threshold).")
    if ab is not None:
        b, d = _row(ab, "Baseline"), _row(ab, "Baseline+dedup")
        if b is not None and d is not None:
            txt += (f"\n\n**Measured effect (ablation):** with duplicates: {int(b['docs_used']):,} docs, leakage "
                    f"{int(b['test_leakage_docs'])}, NB accuracy {b['nb_accuracy']:.3f}, purity {b['km_purity']:.3f}, "
                    f"P@5 {b['precision_at_5']:.3f}; de-duplicated: {int(d['docs_used']):,} docs, leakage "
                    f"{int(d['test_leakage_docs'])}, NB accuracy {d['nb_accuracy']:.3f}, purity "
                    f"{d['km_purity']:.3f}, P@5 {d['precision_at_5']:.3f}. The leakage-free result estimates "
                    "performance without copies shared between training and test sets.")
    return txt


def _q3(p: Pipeline):
    pk, n, _ = p.p_at_k(5, 0.9)
    return (
        "**Approach.** Content-based recommendation in the vector-space model: every article is an L2-normalised "
        "TF-IDF vector; the query (an article or free text) is scored against all documents with cosine similarity "
        "and the Top-K are returned with their scores, metadata and the shared terms that contributed most "
        f"(explanation). With near-duplicate filtering, mean **Precision@5 = {pk:.3f}** over {n} sample queries "
        "(relevant = same desk).\n\n"
        "**Content-based** needs only the document text (plus optional metadata); it works for brand-new articles "
        "(no item cold start), provides explanations, and suits a newsroom/research archive like ours with no user "
        "logs. It has vocabulary mismatch and over-specialisation problems, and does not account for popularity.\n\n"
        "**Collaborative filtering** needs a user-item interaction matrix (reads, clicks, dwell time, ratings). It "
        "finds non-obvious, cross-topic connections ('analysts who read X also read Y') and captures quality/"
        "popularity, but cannot recommend a new article until people interact with it and needs a history for new "
        "users. Prefer it once a large, active user base generates logs; prefer content-based for fresh items, "
        "small user bases and when explanations matter. A hybrid approach is preferable for production.")


def _q4(p: Pipeline):
    rep, ev = p.nb["report"], p.km_eval
    return (
        f"- **Classification** auto-routes each incoming article to the right desk with accuracy "
        f"{rep['accuracy']:.3f}, reducing manual triage and helping analysts manage alerts by desk.\n"
        f"- **Clustering** (purity {ev['purity']:.3f}) organises an unlabelled feed into themes; clusters that do not "
        "match an existing desk signal emerging topics worth a new watch-list.\n"
        f"- **Duplicate detection** collapses {len(p.dups['drop_indices'])} redundant syndicated copies, so volume "
        "metrics ('how much coverage did X get?') are not double-counted.\n"
        "- **Recommendation** retrieves related background coverage for a story in milliseconds and shows the "
        "shared terms behind each result.\n"
        "- **Corpus analytics** (top TF-IDF terms per desk, keyword search over the inverted index) give a quick "
        "picture of what each desk is talking about.\n\n"
        "These methods organise the news stream into a searchable collection with duplicates removed and related "
        "articles linked. This helps analysts prepare briefings faster, miss fewer stories and support client reports with evidence.")


def _q5(p: Pipeline, ab, seeds):
    txt = interpretation(p)
    extra = []
    if seeds is not None:
        extra.append(f"K-Means seed stability (single restart, 5 seeds): purity {seeds['purity'].mean():.3f} ± "
                     f"{seeds['purity'].std(ddof=0):.3f}, RI {seeds['rand_index'].mean():.3f} ± "
                     f"{seeds['rand_index'].std(ddof=0):.3f}. Results depend on initialisation, so the app uses n_init restarts.")
    extra.append(
        "**Limitations.** Bag-of-words ignores word order and meaning; k must be fixed in advance; K-Means assumes "
        "roughly spherical, similarly sized clusters; the BBC corpus is from 2004-05, single-source and only five "
        "broad desks; relevance for recommendation is a same-category proxy; TF-IDF for clustering/recommendation is "
        "fitted on the whole corpus (NB vocabulary is training-only).")
    extra.append(
        "**Improvements for deployment.** Sentence/transformer embeddings to handle synonyms; incremental (online) "
        "index and model updates for a live feed; active learning to label new topics cheaply; hierarchical or "
        "density-based clustering with automatic k; MinHash/LSH for scalable near-duplicate detection; a hybrid "
        "recommender once user interaction logs exist; monitoring for concept drift.")
    return txt + "\n\n" + "\n\n".join(extra)


def render(p: Pipeline):
    header("Inferences & Discussion", "Answers to the five compulsory questions using the current experiment results.")
    ab = st.session_state.get("ablation")
    seeds = st.session_state.get("seed_stability")
    if ab is None:
        c1, c2 = st.columns([3, 1])
        c1.info("Ablation results are not available yet. Answers use the current run only. Run the ablation to "
                "add per-configuration evidence (≈ 1 minute).")
        if c2.button("Run ablation now", key="inf_run_ablation", type="primary"):
            st.session_state.ablation = ab = _run_ablation(p)
    answers = [_q1(p, ab), _q2(p, ab), _q3(p), _q4(p), _q5(p, ab, seeds)]
    for i, (q, a) in enumerate(zip(QUESTIONS, answers), 1):
        section(f"Q{i}. {q}")
        st.markdown(a)
