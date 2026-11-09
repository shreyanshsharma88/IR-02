"""Dashboard page."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from .common import cat_colors, header, kpi_row, section, show, style, truncate
from .state import Pipeline


def render(p: Pipeline):
    header("News Intelligence Dashboard",
           "Text mining & document intelligence for a news-monitoring firm — corpus overview and pipeline status.")

    st.markdown(
        """<div class='context-box'><b>Business context.</b> A media-monitoring / business-research firm ingests a
        continuous stream of unstructured news articles. Analysts need to (1) <b>auto-route</b> each article to the
        right desk (business, politics, tech, ...), (2) <b>discover themes</b> without labels, (3) <b>spot duplicate</b>
        syndicated copies so they are not double-counted, and (4) <b>find related coverage</b> for a story under
        research. This system implements the full pipeline from scratch — inverted index, TF-IDF, cosine similarity,
        Multinomial Naive Bayes, K-Means, Purity / Rand Index and a content-based recommender — on the public
        BBC News corpus (Greene &amp; Cunningham, 2006).</div>""", unsafe_allow_html=True)

    meta, docs = p.meta, p.docs
    total_words = int(meta["n_words"].sum())
    mb = sum(len(t.encode("utf-8")) for t in p.texts_all) / 1e6
    dup_docs = sum(len(g) for g in p.exact_groups)
    kpi_row([
        ("Documents", f"{len(docs):,}", f"Dataset: {p.ds['name']}"),
        ("Categories", f"{meta['category'].nunique()}", ", ".join(sorted(meta['category'].unique()))[:200]),
        ("Total words", f"{total_words:,}"),
        ("Text volume", f"{mb:.2f} MB", "UTF-8 bytes of raw article text"),
        ("Vocabulary", f"{p.tfidf_full.pruned_stats['raw_terms']:,}",
         f"{len(p.tfidf.vocab):,} TF-IDF features kept after pruning"),
        ("Exact-dup groups", f"{len(p.exact_groups):,}", f"{dup_docs} documents involved"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        section("Category distribution")
        cc = p.category_counts().rename_axis("category").reset_index(name="documents")
        fig = px.bar(cc, x="category", y="documents", color="category", text="documents",
                     color_discrete_map=cat_colors(cc["category"]))
        show(style(fig, 340, legend=False), "Articles per desk. Classes are reasonably balanced (sport and business "
             "largest), so accuracy is a meaningful headline metric.")
    with c2:
        section("Document length")
        fig = px.histogram(meta, x="n_words", color="category", nbins=60, barmode="stack",
                           color_discrete_map=cat_colors(meta["category"]))
        fig.update_layout(xaxis_title="words per article", yaxis_title="documents")
        show(style(fig, 340), "Long-tailed length distribution: most articles have 200–600 words, a few exceed 2,000 "
             "(motivates TF-IDF length normalisation).")

    section("Pipeline status", "All stages auto-run with default settings on first load; change them on the "
            "Preprocessing and Text Mining pages.")
    nb, ev = p.nb, p.km_eval
    steps = [
        ("Data ingestion", f"{len(docs):,} documents + separate metadata table ({p.ds['name']})"),
        ("Preprocessing", f"config `{p.cfg.label()}` → {p.proc_stats_all['vocabulary_size']:,} distinct terms"),
        ("Inverted index", f"{len(p.tfidf.index):,} terms with postings lists"),
        ("TF-IDF + L2 normalisation", f"{p.X.shape[0]:,} × {p.X.shape[1]:,} document-term matrix"),
        ("Duplicate detection", f"{len(p.dups['drop_indices'])} redundant docs at cos ≥ {p.dup_threshold:.2f} — "
                                f"{'REMOVED from working set' if p.dedup else 'kept (toggle on Preprocessing page)'}"),
        ("Naive Bayes classifier", f"accuracy {nb['report']['accuracy']:.3f}, macro-F1 {nb['report']['macro']['f1']:.3f}"),
        ("K-Means clustering", f"k={p.km.k}, purity {ev['purity']:.3f}, Rand index {ev['rand_index']:.3f}"),
        ("Recommender", f"content-based Top-K over {p.X.shape[0]:,} documents ready"),
    ]
    rows = "".join(f"<tr><td><span class='status-ok'>✔</span>&nbsp; <b>{s}</b></td><td>{d}</td></tr>" for s, d in steps)
    st.markdown(f"<table style='width:100%'>{rows}</table>", unsafe_allow_html=True)

    section("Sample documents")
    sample = meta.groupby("category", group_keys=False).head(3).copy()
    snip = docs.set_index("doc_id")["text"]
    sample["snippet"] = sample["doc_id"].map(lambda d: truncate(" ".join(str(snip.get(d, "")).split()), 160))
    cols = [c for c in ["doc_id", "title", "category", "source", "period", "n_words", "snippet"] if c in sample]
    st.dataframe(sample[cols], hide_index=True)
