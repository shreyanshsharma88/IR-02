"""Page 5 - Content-based document recommendation."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from textmining.preprocessing import preprocess

from .common import cat_colors, header, kpi_row, section, show, style, truncate
from .state import Pipeline


def render(p: Pipeline):
    header("Document Recommendation",
           "“Analysts who are reading this story should also read …” — content-based Top-K via cosine similarity.")
    rec = p.recommender
    meta = p.meta_w
    c1, c2, c3 = st.columns([1.1, 1, 1])
    mode = c1.radio("Query", ["Pick a document", "Free text"], horizontal=True, key="rec_mode")
    k = c2.slider("Top-K", 1, 20, 5, key="rec_k")
    hide = c3.checkbox("Hide near-duplicates of the query", True, key="rec_hide",
                       help="Skip results whose cosine with the query is ≥ the threshold (syndicated copies).")
    thr = c3.slider("Duplicate threshold", 0.70, 1.00, 0.90, 0.01, key="rec_thr", disabled=not hide)
    dup_thr = thr if hide else None

    if mode == "Pick a document":
        opts = p.doc_options()
        default = next((i for i, o in enumerate(opts) if "Greenspan" in o), 0)
        q = st.selectbox("Search a document by id or title (type to filter)", range(len(opts)), index=default,
                         format_func=lambda i: opts[i], key="rec_doc")
        qvec = p.X[q]
        results = rec.similar_to_doc(q, k, dup_thr)
        q_label = meta["category"].iat[q]
        snippet = truncate(" ".join(p.texts[q].split()), 450).replace("$", "&#36;")
        st.markdown(f"<div class='doc-card'><b>{meta['title'].iat[q]}</b> &nbsp;·&nbsp; {meta['doc_id'].iat[q]} "
                    f"&nbsp;·&nbsp; <i>{q_label}</i> &nbsp;·&nbsp; {meta['source'].iat[q]} {meta['period'].iat[q]}"
                    f"<br><span style='color:#6b7280'>{snippet}</span></div>", unsafe_allow_html=True)
        skipped = int(((p.X @ qvec) >= thr).sum()) - 1 if hide else 0
    else:
        txt = st.text_area("Describe what you are researching", "Mobile phone makers are racing to add music "
                           "players and cameras to new handsets.", key="rec_text", height=90)
        toks = preprocess(txt, p.cfg)
        qvec = p.tfidf.transform([toks])[0] if toks else None
        results = rec.similar_to_text(toks, k) if toks else []
        q_label, skipped = None, 0
        st.caption(f"Query after preprocessing: `{' '.join(toks[:40])}`")

    if not results:
        st.warning("No recommendations — the query shares no terms with the indexed vocabulary.")
        return

    rows = []
    for rank, (i, score) in enumerate(results, 1):
        why = rec.explain(qvec, i, 6)
        rows.append({"rank": rank, "score": round(score, 4), "doc_id": meta["doc_id"].iat[i],
                     "title": meta["title"].iat[i], "category": meta["category"].iat[i],
                     "source": meta["source"].iat[i], "period": meta["period"].iat[i],
                     "top shared terms": ", ".join(f"{t} ({w:.3f})" for t, w in why)})
    res = pd.DataFrame(rows)

    section(f"Top-{k} recommendations")
    p_q = (res["category"] == q_label).mean() if q_label is not None else None
    kpis = [("Best score", f"{res['score'].max():.4f}"), ("Mean score", f"{res['score'].mean():.4f}")]
    if p_q is not None:
        kpis.append((f"P@{k} (this query)", f"{p_q:.2f}", "share of results in the query's category"))
        if hide:
            kpis.append(("Dups hidden", f"{max(skipped, 0)}", "near-duplicates of the query skipped"))
    pk, n_sample, _ = p.p_at_k(k, dup_thr)
    kpis.append((f"Mean P@{k} (sample)", f"{pk:.3f}", f"averaged over {n_sample} query documents, "
                 "same-category relevance proxy"))
    kpi_row(kpis)
    st.dataframe(res, hide_index=True, column_config={
        "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.4f")})

    c1, c2 = st.columns([1.4, 1])
    with c1:
        bar_df = res.assign(label=res["rank"].astype(str) + ". " + res["title"].map(lambda s: truncate(s, 45)))
        fig = px.bar(bar_df, x="score", y="label", orientation="h", color="category", text="score",
                     color_discrete_map=cat_colors(p.labels), hover_data=["doc_id", "top shared terms"])
        fig.update_yaxes(autorange="reversed", title="", categoryorder="array", categoryarray=bar_df["label"].tolist())
        fig.update_xaxes(range=[0, 1])
        show(style(fig, max(260, 46 * len(res)), "Cosine similarity of recommended documents"),
             "Colour = desk of the recommended article. Scores fall off quickly after the first few results.")
    with c2:
        sweep = []
        for kk in (1, 3, 5, 10, 20):
            v, _, _ = p.p_at_k(kk, dup_thr)
            sweep.append({"K": kk, "Precision@K": v})
        sw = pd.DataFrame(sweep)
        fig = px.line(sw, x="K", y="Precision@K", markers=True, color_discrete_sequence=["#2E5EAA"])
        fig.update_yaxes(range=[0, 1.02])
        show(style(fig, 300, "Mean Precision@K over a ~300-doc sample", legend=False),
             "Relevance proxy: a recommendation counts as relevant if it shares the query's category.")

    with st.expander("How it works"):
        st.markdown(
            "Every article is a vector in the **vector-space model** (TF-IDF, `ltc` weighting, L2-normalised). The "
            "query (a chosen article or free text processed by the same pipeline) is scored against all documents:")
        st.latex(r"\text{score}(q,d)=\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\lVert\vec d\rVert}"
                 r"=\sum_t \hat w_{t,q}\,\hat w_{t,d}")
        st.markdown(
            "Documents are ranked by score and the Top-K returned (excluding the query itself and, optionally, "
            "near-duplicates with score ≥ threshold). **Why recommended** lists the terms with the largest "
            "contribution ŵ(t,q)·ŵ(t,d) to the dot product — a built-in explanation.")
    with st.expander("Limitations"):
        st.markdown(
            "- **Vocabulary mismatch** — synonyms (`mobile` vs `cellphone`) get zero similarity; bag-of-words has no "
            "semantics or word order.\n"
            "- **No understanding of meaning / stance** — two articles with opposite conclusions on the same topic "
            "score high.\n"
            "- **Duplicate flooding** — syndicated copies score ≈1.0 and crowd out new information (hence the "
            "near-duplicate filter).\n"
            "- **Popularity- and recency-blind** — ignores what analysts actually read or how fresh a story is.\n"
            "- **Over-specialisation** — recommends more of the same topic; little serendipity.\n"
            "- **Proxy evaluation** — Precision@K uses same-category as relevance, which is only a rough stand-in for "
            "real analyst judgements.")
    with st.expander("Content-based vs collaborative filtering"):
        st.dataframe(pd.DataFrame([
            {"aspect": "Input data", "content-based (this system)": "Document text only",
             "collaborative filtering": "User–item interaction logs (reads, clicks, ratings)"},
            {"aspect": "New / cold-start documents", "content-based (this system)": "Recommendable immediately",
             "collaborative filtering": "Cannot recommend until users interact with it"},
            {"aspect": "New users", "content-based (this system)": "Works from a single query document",
             "collaborative filtering": "Needs a usage history"},
            {"aspect": "Explainability", "content-based (this system)": "High — shared terms",
             "collaborative filtering": "Low — 'similar users read this'"},
            {"aspect": "Serendipity", "content-based (this system)": "Low — same topic",
             "collaborative filtering": "High — cross-topic discovery"},
            {"aspect": "Feasible here?", "content-based (this system)": "Yes",
             "collaborative filtering": "No — the corpus has no user interaction data"},
        ]), hide_index=True)
        st.caption("In production a hybrid (content-based for cold start + collaborative once logs accumulate) "
                   "usually works best.")
