"""Page 3 - Preprocessing, inverted index, TF-IDF, cosine similarity, duplicates."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from textmining.preprocessing import PreprocessConfig, preprocess, raw_tokens

from .common import AFTER_COLOR, BEFORE_COLOR, header, heatmap, kpi_row, section, show, style, truncate
from .state import Pipeline, persist


# ------------------------------------------------------------------ config form
def _config_form(p: Pipeline):
    cfg, tp = p.cfg, p.tfidf_params
    with st.form("pp_form"):
        st.markdown("**Preprocessing pipeline**")
        c1, c2, c3, c4 = st.columns(4)
        strip_markup = c1.checkbox("Strip HTML / URLs", cfg.strip_markup)
        lowercase = c1.checkbox("Lowercase", cfg.lowercase)
        remove_numbers = c2.checkbox("Remove numbers", cfg.remove_numbers)
        remove_punct = c2.checkbox("Remove punctuation", cfg.remove_punctuation)
        remove_stop = c3.checkbox("Remove stop-words", cfg.remove_stopwords)
        norm_opts = ["none", "stem", "lemma"]
        normalizer = c3.radio("Normaliser", norm_opts, index=norm_opts.index(cfg.normalizer), horizontal=True,
                              help="stem = Porter stemmer, lemma = WordNet lemmatiser (NLTK used only for these)")
        min_len = c4.number_input("Min token length", 1, 10, cfg.min_token_len)
        ngram = c4.slider("n-gram range", 1, 3, (cfg.ngram_min, cfg.ngram_max))
        custom = st.text_input("Custom stop-words (comma separated)", ", ".join(cfg.custom_stopwords),
                               placeholder="e.g. bbc, people, year")
        st.markdown("**TF-IDF feature space**")
        d1, d2, d3, d4 = st.columns(4)
        min_df = d1.number_input("min_df (documents)", 1, 50, int(tp["min_df"]))
        max_df = d2.slider("max_df (ratio of docs)", 0.3, 1.0, float(tp["max_df"]), 0.05)
        max_feat = d3.number_input("max_features (0 = no cap)", 0, 50000, int(tp["max_features"] or 0), 500)
        sublinear = d4.checkbox("Sublinear tf (1 + log10 tf)", bool(tp["sublinear"]))
        submitted = st.form_submit_button("Apply & rebuild pipeline", type="primary")
    if submitted:
        st.session_state.cfg = PreprocessConfig(
            strip_markup=strip_markup, lowercase=lowercase, remove_numbers=remove_numbers,
            remove_punctuation=remove_punct, remove_stopwords=remove_stop, normalizer=normalizer,
            min_token_len=int(min_len), ngram_min=int(ngram[0]), ngram_max=int(ngram[1]),
            custom_stopwords=[w.strip() for w in custom.split(",") if w.strip()])
        st.session_state.tfidf_params = dict(min_df=int(min_df), max_df=float(max_df),
                                             max_features=int(max_feat), sublinear=bool(sublinear))
        st.session_state.pop("ablation", None)
        st.rerun()


# ------------------------------------------------------------------ before / after
def _zipf_points(tf_counter, n_points=400):
    freqs = np.array(sorted(tf_counter.values(), reverse=True), dtype=float)
    if len(freqs) == 0:
        return np.array([]), np.array([])
    ranks = np.unique(np.logspace(0, math.log10(len(freqs)), n_points).astype(int))
    return ranks, freqs[ranks - 1]


def _before_after(p: Pipeline):
    rs, ps = p.raw_stats, p.proc_stats_all
    keys = ["documents", "total_tokens", "vocabulary_size", "avg_tokens_per_doc", "min_tokens", "max_tokens",
            "type_token_ratio", "hapax_legomena"]
    def _f(v):
        return f"{v:,}" if isinstance(v, int) else f"{v:,.4g}" if v < 1 else f"{v:,.1f}"

    def _chg(b, a):
        return f"{(a - b) / b * 100:+.1f}%" if b else ""

    tbl = pd.DataFrame({"statistic": [k.replace("_", " ") for k in keys],
                        "before (raw tokens)": [_f(rs[k]) for k in keys],
                        "after preprocessing": [_f(ps[k]) for k in keys],
                        "change": [_chg(rs[k], ps[k]) for k in keys]})
    kpi_row([
        ("Vocabulary before", f"{rs['vocabulary_size']:,}"),
        ("Vocabulary after", f"{ps['vocabulary_size']:,}",
         "distinct terms after preprocessing (before TF-IDF pruning)"),
        ("Reduction", f"{(1 - ps['vocabulary_size'] / max(rs['vocabulary_size'], 1)) * 100:.1f}%"),
        ("Tokens / doc", f"{rs['avg_tokens_per_doc']:.0f} → {ps['avg_tokens_per_doc']:.0f}"),
    ])
    st.dataframe(tbl, hide_index=True)

    c1, c2 = st.columns(2)
    for col, stats, name, color in ((c1, rs, "Before (raw tokens)", BEFORE_COLOR),
                                    (c2, ps, "After preprocessing", AFTER_COLOR)):
        top = pd.DataFrame(stats["term_freq"].most_common(20), columns=["term", "frequency"])
        fig = px.bar(top, x="frequency", y="term", orientation="h", color_discrete_sequence=[color])
        fig.update_yaxes(autorange="reversed", title="")
        with col:
            show(style(fig, 470, f"Top-20 terms — {name}", legend=False))
    st.caption("Before: dominated by punctuation and function words (the, ',', '.', to, of). After: content-bearing "
               "stems (year, market, govern, game ...) — exactly what TF-IDF and the classifiers need.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for stats, name, color in ((rs, "before", BEFORE_COLOR), (ps, "after", AFTER_COLOR)):
            r, f = _zipf_points(stats["term_freq"])
            fig.add_trace(go.Scatter(x=r, y=f, mode="lines+markers", name=name, marker=dict(size=3, color=color),
                                     line=dict(color=color)))
        fig.update_xaxes(type="log", title="rank (log)")
        fig.update_yaxes(type="log", title="frequency (log)")
        show(style(fig, 380, "Zipf rank–frequency (log-log)"),
             "Both curves are near-linear on log-log axes (Zipf's law). Preprocessing flattens the head (stop-words "
             "and punctuation removed) and shortens the tail (stemming merges variants).")
    with c2:
        dl = pd.DataFrame({"tokens per document": rs["lengths"] + ps["lengths"],
                           "stage": ["before"] * len(rs["lengths"]) + ["after"] * len(ps["lengths"])})
        fig = px.histogram(dl, x="tokens per document", color="stage", nbins=80, barmode="overlay", opacity=0.7,
                           color_discrete_map={"before": BEFORE_COLOR, "after": AFTER_COLOR})
        show(style(fig, 380, "Document length before vs after"),
             "Preprocessing roughly halves document length — less noise per document, faster indexing.")


def _live_demo(p: Pipeline):
    opts = p.meta["doc_id"].astype(str) + " — " + p.meta["title"]
    choice = st.selectbox("Document", range(len(opts)), format_func=lambda i: opts.iloc[i], key="pp_demo_doc")
    text = p.texts_all[choice]
    raw = raw_tokens(text)
    proc = preprocess(text, p.cfg)
    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown("**Raw text**")
    safe = truncate(text, 900).replace("$", "&#36;")
    c1.markdown(f"<div class='doc-card'>{safe}</div>", unsafe_allow_html=True)
    c2.markdown(f"**Raw tokens** ({len(raw)})")
    c2.code(" | ".join(raw[:90]), language=None, wrap_lines=True)
    c3.markdown(f"**Processed tokens** ({len(proc)})")
    c3.code(" | ".join(proc[:90]), language=None, wrap_lines=True)


# ------------------------------------------------------------------ TF-IDF
def _tfidf_section(p: Pipeline):
    m = p.tfidf
    st.latex(r"w_{t,d} = \underbrace{(1+\log_{10} tf_{t,d})}_{\text{sublinear tf}} \times "
             r"\underbrace{\log_{10}\frac{N}{df_t}}_{\text{idf}} \qquad "
             r"\hat w_{t,d} = \frac{w_{t,d}}{\sqrt{\sum_{t'} w_{t',d}^2}}")
    st.caption(f"SMART `ltc` weighting, computed term-at-a-time from the inverted index's postings lists; "
               f"N = {m.index.n_docs:,} documents. Rows are L2 (cosine) length-normalised.")
    ps = m.pruned_stats
    kpi_row([("Terms in index", f"{ps['raw_terms']:,}"),
             ("Removed: rare (< min_df)", f"{ps['removed_rare(<min_df)']:,}"),
             ("Removed: too common (> max_df)", f"{ps['removed_common(>max_df)']:,}"),
             ("Kept features", f"{ps['kept_terms']:,}", "after max_features cap (most frequent kept)"),
             ("Matrix density", f"{(m.matrix > 0).mean() * 100:.2f}%")])


def _index_explorer(p: Pipeline):
    m = p.tfidf
    c1, c2 = st.columns([1, 2])
    word = c1.text_input("Look up a word (it is preprocessed with the current config)", "economy", key="ix_word")
    toks = preprocess(word, p.cfg) if word.strip() else []
    term = toks[0] if toks else word.strip().lower()
    post = m.index.lookup(term)
    df_t = len(post)
    N = m.index.n_docs
    in_vocab = term in m.term_to_col
    idf = m.idf[term] if in_vocab else (math.log10(N / df_t) if df_t else 0.0)
    with c1:
        st.markdown(f"Index term: `{term}`")
        kpi_row([("df", f"{df_t:,}"), ("idf", f"{idf:.3f}")], border=True)
        if df_t and not in_vocab:
            st.warning("Term is in the inverted index but was pruned from the TF-IDF feature space (min_df/max_df/"
                       "max_features).")
        elif not df_t:
            st.info("Term not found in the index.")
    with c2:
        if df_t:
            col = m.term_to_col.get(term)
            rows = []
            for d, tf in post.items():
                rows.append({"doc_id": p.meta_w["doc_id"].iat[d], "title": p.meta_w["title"].iat[d],
                             "category": p.meta_w["category"].iat[d], "tf": tf,
                             "w = (1+log tf)·idf": round((1 + math.log10(tf)) * idf, 4),
                             "tf-idf (normalised)": round(float(m.matrix[d, col]), 4) if col is not None else None})
            tbl = pd.DataFrame(rows).sort_values(["tf", "doc_id"], ascending=[False, True])
            st.markdown(f"**Postings list** — {df_t:,} documents")
            st.dataframe(tbl, hide_index=True, height=300)


def _idf_and_doc_terms(p: Pipeline):
    m = p.tfidf
    c1, c2 = st.columns(2)
    with c1:
        idf = pd.DataFrame(m.idf_table(), columns=["term", "df", "idf"])
        fig = px.histogram(idf, x="idf", nbins=50, color_discrete_sequence=[AFTER_COLOR])
        fig.update_layout(yaxis_title="number of terms")
        show(style(fig, 380, "IDF distribution of kept features", legend=False),
             "Most features are fairly rare (high idf); the left tail holds broad terms present in many articles.")
        with st.expander("Lowest / highest idf terms"):
            a, b = st.columns(2)
            a.dataframe(idf.tail(10).iloc[::-1].round(3), hide_index=True)
            b.dataframe(idf.head(10).round(3), hide_index=True)
    with c2:
        opts = p.doc_options()
        i = st.selectbox("Document", range(len(opts)), format_func=lambda k: opts[k], key="pp_topterms_doc")
        top = pd.DataFrame(m.top_terms(i, 15), columns=["term", "weight"])
        fig = px.bar(top, x="weight", y="term", orientation="h", color_discrete_sequence=[AFTER_COLOR])
        fig.update_yaxes(autorange="reversed", title="")
        show(style(fig, 380, "Top TF-IDF terms of the selected document", legend=False),
             f"Category: {p.meta_w['category'].iat[i]}. High weight = frequent in this article but rare corpus-wide.")


def _cosine(p: Pipeline):
    opts = p.doc_options()
    X = p.X
    c1, c2 = st.columns(2)
    a = c1.selectbox("Document A", range(len(opts)), format_func=lambda k: opts[k], key="cos_a")
    nn = p.recommender.similar_to_doc(a, 1)
    default_b = nn[0][0] if nn else min(1, len(opts) - 1)
    b = c2.selectbox("Document B", range(len(opts)), index=default_b, format_func=lambda k: opts[k], key="cos_b")
    va, vb = X[a], X[b]
    dot = float(va @ vb)
    na, nb = float(np.sqrt(va @ va)), float(np.sqrt(vb @ vb))
    cos = dot / (na * nb) if na and nb else 0.0
    st.latex(r"\cos(\vec a,\vec b)=\frac{\vec a\cdot\vec b}{\lVert\vec a\rVert\,\lVert\vec b\rVert}"
             rf"=\frac{{{dot:.4f}}}{{{na:.3f}\times{nb:.3f}}}={cos:.4f}")
    contrib = va * vb
    idx = np.argsort(-contrib)[:12]
    shared = pd.DataFrame([{"term": p.tfidf.vocab[k], "w(A)": round(float(va[k]), 4), "w(B)": round(float(vb[k]), 4),
                            "contribution w(A)·w(B)": round(float(contrib[k]), 4),
                            "share of cosine": f"{contrib[k] / dot * 100:.1f}%" if dot else "-"}
                           for k in idx if contrib[k] > 0])
    c1, c2 = st.columns([1, 1])
    with c1:
        kpi_row([("Cosine similarity", f"{cos:.4f}"), ("Shared terms", f"{int(((va > 0) & (vb > 0)).sum())}"),
                 ("Same category", "Yes" if p.labels[a] == p.labels[b] else "No")])
        st.caption("Document B defaults to A's nearest neighbour. Vectors are already unit length, so the cosine "
                   "equals the dot product.")
    with c2:
        if len(shared):
            st.dataframe(shared, hide_index=True, height=280)
        else:
            st.info("No shared terms — the documents are orthogonal (cosine 0).")

    sample = p.sample_balanced(4)
    S = X[sample] @ X[sample].T
    labels = [f"{p.meta_w['category'].iat[i][:4]} · {truncate(p.meta_w['title'].iat[i], 28)}" for i in sample]
    fig = heatmap(np.round(S, 2).tolist(), labels, labels, height=620, fmt=".2f", colorscale="Blues")
    fig.update_xaxes(tickangle=45)
    show(fig, "Cosine heatmap of 20 sampled articles (4 per category, grouped). Bright diagonal blocks show that "
              "articles from the same desk share vocabulary — the signal that NB, K-Means and the recommender exploit.")


def _duplicates(p: Pipeline):
    ss = st.session_state
    c1, c2 = st.columns([1, 1])
    c1.slider("Near-duplicate cosine threshold", 0.70, 1.00, float(ss.dup_threshold), 0.01, key="w_dup_threshold",
              on_change=persist, args=("w_dup_threshold", "dup_threshold"))
    c2.checkbox("Remove duplicates before modelling", value=bool(ss.dedup), key="w_dedup", on_change=persist,
                args=("w_dedup", "dedup"),
                help="Keeps the first document of every exact/near-duplicate group; the reduced working set is used "
                     "by Naive Bayes, K-Means, evaluation and recommendation.")
    rep = p.dups
    pairs = pd.DataFrame(rep["pairs"])
    n_exact = int((pairs["type"] == "exact").sum()) if len(pairs) else 0
    kpi_row([("Exact-dup groups", f"{len(rep['exact_groups'])}"), ("Pairs ≥ threshold", f"{len(pairs):,}"),
             ("Exact pairs", f"{n_exact:,}"), ("Near (non-identical) pairs", f"{len(pairs) - n_exact:,}"),
             ("Docs dropped if dedup", f"{len(rep['drop_indices'])}"),
             ("Working set", f"{len(p.work_idx):,} docs", "ON: deduplicated" if p.dedup else "OFF: all documents")])
    if len(pairs):
        meta = p.meta
        pairs["doc A"] = meta["doc_id"].values[pairs["i"]]
        pairs["doc B"] = meta["doc_id"].values[pairs["j"]]
        pairs["title A"] = meta["title"].values[pairs["i"]]
        pairs["title B"] = meta["title"].values[pairs["j"]]
        pairs["cat A"] = meta["category"].values[pairs["i"]]
        pairs["cat B"] = meta["category"].values[pairs["j"]]
        pairs["cosine"] = pairs["cosine"].round(4)
        pairs["jaccard_5shingle"] = pairs["jaccard_5shingle"].round(4)
        show_type = st.radio("Show", ["all", "near only", "exact only"], horizontal=True, key="dup_filter")
        view = pairs if show_type == "all" else pairs[pairs["type"] == ("near" if show_type == "near only" else "exact")]
        st.dataframe(view.sort_values("cosine", ascending=False)[
            ["doc A", "doc B", "type", "cosine", "jaccard_5shingle", "title A", "title B", "cat A", "cat B"]],
            hide_index=True, height=300)
        st.caption("Cosine on TF-IDF vectors catches re-worded copies; the 5-word shingle Jaccard confirms how much "
                   "literal text is shared (1.0 = identical wording).")


def render(p: Pipeline):
    header("Preprocessing & Feature Engineering",
           "Configurable cleaning pipeline → manual inverted index → TF-IDF with cosine normalisation.")
    with st.expander("Pipeline configuration", expanded=False):
        _config_form(p)
    st.caption(f"Active config: `{p.cfg.label()}` · min_df={p.tfidf_params['min_df']}, "
               f"max_df={p.tfidf_params['max_df']}, max_features={p.tfidf_params['max_features'] or 'none'}, "
               f"sublinear={p.tfidf_params['sublinear']}")

    tabs = st.tabs(["Before vs after", "Live demo", "TF-IDF & inverted index", "Cosine similarity",
                    "Duplicates"])
    with tabs[0]:
        section("Corpus statistics before vs after preprocessing")
        _before_after(p)
    with tabs[1]:
        section("One document through the pipeline")
        _live_demo(p)
    with tabs[2]:
        section("TF-IDF weighting")
        _tfidf_section(p)
        section("Inverted-index explorer", "term → postings {doc: tf}; built by hand in `textmining/tfidf.py`.")
        _index_explorer(p)
        section("IDF distribution & document top terms")
        _idf_and_doc_terms(p)
    with tabs[3]:
        section("Cosine similarity between two documents")
        _cosine(p)
    with tabs[4]:
        section("Exact & near-duplicate detection")
        _duplicates(p)
