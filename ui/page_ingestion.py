"""Data ingestion page."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from .common import cat_colors, header, kpi_row, section, show, style, truncate
from .state import Pipeline, default_dataset, load_upload, set_dataset

NOISE_PATTERNS = {
    "Capitalised / proper nouns": r"\b[A-Z][a-z]+\b",
    "ALL-CAPS tokens": r"\b[A-Z]{2,}\b",
    "Numbers & figures": r"\b\d[\d,.]*\b",
    "Currency & % symbols": r"[£$€%]",
    "Quotes & apostrophes": r"[\"'`“”‘’]",
    "Other punctuation": r"[.,;:!?()\-–—/]",
    "Hyphenated words": r"\b\w+-\w+\b",
}


@st.cache_resource(show_spinner=False, max_entries=4)
def _noise(ds_key, _texts):
    out = []
    for name, pat in NOISE_PATTERNS.items():
        rx = re.compile(pat)
        total, examples, ctx = 0, Counter(), ""
        for t in _texts:
            found = rx.findall(t)
            total += len(found)
            if len(examples) < 400:
                examples.update(found)
            if not ctx and found:
                m = rx.search(t, 50) or rx.search(t)
                if m:
                    s = max(0, m.start() - 60)
                    ctx = " ".join(t[s:m.end() + 60].split())
                    ctx = ctx.replace(m.group(0), f"**{m.group(0)}**", 1) if m.group(0).strip("*") else ctx
        out.append({"noise type": name, "occurrences": total,
                    "examples": ", ".join(e for e, _ in examples.most_common(8)), "context": ctx})
    return pd.DataFrame(out)


def _load_controls():
    section("1 · Choose a data source")
    src = st.radio("Data source", ["Default BBC News corpus", "Upload CSV / JSON / ZIP of .txt"],
                   horizontal=True, key="ing_source")
    if src.startswith("Default"):
        st.caption("2,225 BBC News articles (2004–05), 5 desks. Text in `data/documents.csv`, metadata in "
                   "`data/metadata.csv`.")
        if st.session_state.ds["key"] != "bbc-default":
            if st.button("Load default BBC dataset", type="primary"):
                set_dataset(default_dataset())
                st.rerun()
        else:
            st.success("Default BBC News corpus is loaded.")
    else:
        st.caption("CSV/JSON need a text column (`text`, `content`, `body`, ...); optional `doc_id`, `title`, "
                   "`category`/`label`. ZIP: one `.txt` per document, folder name = category, first line = title. "
                   "Missing ids are auto-assigned; duplicate ids are made unique.")
        up = st.file_uploader("Upload dataset", type=["csv", "json", "jsonl", "zip"])
        if up is not None and st.button("Ingest uploaded file", type="primary"):
            try:
                ds = load_upload(up.name, up.getvalue())
                if len(ds["docs"]) < 5:
                    st.error("Need at least 5 documents.")
                else:
                    set_dataset(ds)
                    st.success(f"Loaded {len(ds['docs'])} documents from {up.name}.")
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not parse file: {e}")


def render(p: Pipeline):
    header("Data Ingestion", "Load a corpus, keep raw text and metadata logically separate, and profile it.")
    _load_controls()
    docs, meta = p.docs, p.meta

    section("2 · Stored representation", "Document text and metadata are two tables joined only by `doc_id`.")
    t1, t2 = st.tabs([f"Documents table ({len(docs):,} rows × {docs.shape[1]} cols)",
                      f"Metadata table ({len(meta):,} rows × {meta.shape[1]} cols)"])
    with t1:
        view = docs.head(300).copy()
        view["text"] = view["text"].map(lambda s: truncate(" ".join(str(s).split()), 220))
        st.dataframe(view, hide_index=True, height=320)
    with t2:
        st.dataframe(meta.head(300), hide_index=True, height=320)

    section("3 · Corpus statistics")
    ids_docs_unique = docs["doc_id"].is_unique
    ids_meta_unique = meta["doc_id"].is_unique
    ids_match = set(docs["doc_id"].astype(str)) == set(meta["doc_id"].astype(str))
    kpi_row([
        ("Documents", f"{len(docs):,}"),
        ("Characters", f"{int(meta['n_chars'].sum()):,}"),
        ("Words", f"{int(meta['n_words'].sum()):,}"),
        ("Avg words / doc", f"{meta['n_words'].mean():.0f}"),
        ("Unique doc_ids", "Yes" if ids_docs_unique and ids_meta_unique else "No",
         "doc_id is unique in both tables"),
        ("Text ↔ metadata ids match", "Yes" if ids_match else "No"),
    ])
    c1, c2 = st.columns([1, 1])
    with c1:
        cc = meta.groupby("category").agg(documents=("doc_id", "count"), avg_words=("n_words", "mean"),
                                          avg_chars=("n_chars", "mean")).round(1).reset_index()
        fig = px.bar(cc, x="category", y="documents", color="category", text="documents",
                     color_discrete_map=cat_colors(cc["category"]))
        show(style(fig, 300, legend=False), "Documents per category after ingestion.")
        st.dataframe(cc, hide_index=True)
    with c2:
        desc = meta[["n_chars", "n_words"]].describe().T.round(1)
        desc.index = ["characters / doc", "words / doc"]
        st.markdown("**Length statistics**")
        st.dataframe(desc)
        mv = pd.concat([docs.isna().sum().rename("missing").to_frame().assign(table="documents"),
                        meta.isna().sum().rename("missing").to_frame().assign(table="metadata")])
        mv["empty strings"] = [int((docs[c].astype(str).str.strip() == "").sum()) if t == "documents"
                               else int((meta[c].astype(str).str.strip() == "").sum())
                               for c, t in zip(mv.index, mv["table"])]
        st.markdown("**Missing values per column**")
        st.dataframe(mv.reset_index(names="column")[["table", "column", "missing", "empty strings"]],
                     hide_index=True, height=260)

    section("4 · Exact duplicate preview", "MD5 hash of whitespace/case-normalised text; full exact + near-duplicate "
            "analysis lives on the Preprocessing page.")
    groups = p.exact_groups
    involved = sum(len(g) for g in groups)
    kpi_row([("Exact-duplicate groups", f"{len(groups)}"), ("Documents involved", f"{involved}"),
             ("Redundant copies", f"{involved - len(groups)}", "documents beyond the first in each group")])
    if groups:
        rows = []
        for g in groups[:40]:
            m = meta.iloc[g]
            rows.append({"group size": len(g), "doc_ids": ", ".join(m["doc_id"].astype(str)),
                         "title": m["title"].iloc[0], "categories": ", ".join(sorted(set(m["category"])))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=260)

    section("5 · Noise in the raw text", "Why preprocessing is needed: casing, punctuation, numbers and symbols "
            "inflate the vocabulary and split identical concepts into several tokens.")
    noise = _noise(p.key, p.texts_all)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = px.bar(noise, x="occurrences", y="noise type", orientation="h", text="occurrences",
                     color_discrete_sequence=["#C8423B"])
        fig.update_yaxes(categoryorder="total ascending", title="")
        show(style(fig, 330, legend=False), "Occurrences of each noise type in the raw corpus.")
    with c2:
        st.dataframe(noise[["noise type", "examples"]], hide_index=True, height=330)
    with st.expander("Context examples (match in bold)"):
        for _, r in noise.iterrows():
            st.markdown(f"- **{r['noise type']}** — …{r['context']}…".replace("$", "\\$"))
    raw_tf = p.raw_stats["term_freq"]
    variants = [w for w in ["The", "the", "THE", "Government", "government", "said", "Said"] if w in raw_tf]
    if variants:
        st.caption("Case variants counted as different raw tokens: " +
                   ", ".join(f"`{w}` × {raw_tf[w]:,}" for w in variants))
