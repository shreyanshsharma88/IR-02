"""Performance analytics page."""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st

from textmining import metrics as M
from textmining.kmeans import KMeans
from textmining.pipeline import run_experiment
from textmining.preprocessing import PreprocessConfig

from .common import PALETTE, header, kpi_row, section, show, style
from .state import Pipeline

ABLATIONS = [
    ("Baseline", dict(), False),
    ("No stop-word removal", dict(remove_stopwords=False), False),
    ("Lemma", dict(normalizer="lemma"), False),
    ("No normalizer", dict(normalizer="none"), False),
    ("Uni+bigram", dict(ngram_max=2), False),
    ("No lowercasing", dict(lowercase=False), False),
    ("Baseline+dedup", dict(), True),
]


def _run_ablation(p: Pipeline) -> pd.DataFrame:
    tp, prm = p.tfidf_params, p.km_params
    rows = []
    prog = st.progress(0.0, text="Running ablation ...")
    # reuse cached tokens when the baseline config is active
    baseline_tokens = p.tokens_all if p.cfg == PreprocessConfig() else None
    for n, (name, change, dedup) in enumerate(ABLATIONS):
        prog.progress(n / len(ABLATIONS), text=f"Running '{name}' ({n + 1}/{len(ABLATIONS)}) ...")
        cfg = PreprocessConfig(**change)
        t0 = time.perf_counter()
        res = run_experiment(p.docs, p.meta, cfg, min_df=tp["min_df"], max_df=tp["max_df"],
                             max_features=tp["max_features"] or None, sublinear=tp["sublinear"], dedup=dedup,
                             dup_threshold=p.dup_threshold, k=prm["k"], seed=prm["seed"],
                             test_size=p.nb_params["test_size"], alpha=p.nb_params["alpha"], init=prm["init"],
                             n_init=prm["n_init"], p_at_k=5,
                             tokens=baseline_tokens if not change else None)
        timings = res.pop("timings")
        res = {"run": name, **res, "total_s": time.perf_counter() - t0,
               **{f"t_{k}": v for k, v in timings.items()}}
        rows.append(res)
    prog.progress(1.0, text="Done")
    return pd.DataFrame(rows)


def _seed_stability(p: Pipeline) -> pd.DataFrame:
    prm = p.km_params
    rows = []
    for s in (1, 7, 42, 123, 2024):
        km = KMeans(k=prm["k"], init=prm["init"], max_iter=prm["max_iter"], tol=prm["tol"], n_init=1, seed=s).fit(p.X)
        cl = km.labels_.tolist()
        rows.append({"seed": s, "purity": M.purity(cl, p.labels), "rand_index": M.rand_index(cl, p.labels),
                     "SSE": km.sse_, "iterations": km.n_iter_})
    return pd.DataFrame(rows)


def render(p: Pipeline):
    header("Performance Analytics", "Stage timings, preprocessing comparisons and K-Means seed stability.")
    section("Stage timings for the current run")
    t = p.timings()
    tdf = pd.DataFrame({"stage": list(t), "seconds": list(t.values())})
    kpi_row([("Total pipeline", f"{tdf['seconds'].sum():.2f} s"), ("Documents", f"{len(p.work_idx):,}"),
             ("Features", f"{p.X.shape[1]:,}"),
             ("Throughput (preprocessing)", f"{len(p.texts_all) / max(t['Preprocessing'], 1e-9):,.0f} docs/s")])
    fig = px.bar(tdf, x="seconds", y="stage", orientation="h", text=tdf["seconds"].map(lambda v: f"{v:.2f}s"),
                 color_discrete_sequence=["#2E5EAA"])
    fig.update_yaxes(autorange="reversed", title="")
    show(style(fig, 320, legend=False), "Elapsed time for each stage's latest computation. Results are cached per "
                                        "configuration and reused when you revisit a page.")

    section("Preprocessing ablation", "Runs the full pipeline (preprocess → TF-IDF → NB → K-Means → P@5) for each "
            "configuration with the current TF-IDF / model settings. Takes about a minute.")
    if st.button("Run ablation (7 configurations)", type="primary", key="run_ablation"):
        st.session_state.ablation = _run_ablation(p)
    if "ablation" in st.session_state:
        ab = st.session_state.ablation
        show_cols = ["run", "docs_used", "dropped_duplicates", "vocab_raw", "vocab_tfidf", "avg_tokens", "nb_accuracy",
                     "nb_macro_f1", "test_leakage_docs", "km_purity", "km_rand_index", "km_iterations",
                     "precision_at_5", "total_s"]
        st.dataframe(ab[show_cols].style.format({c: "{:.4f}" for c in ["nb_accuracy", "nb_macro_f1", "km_purity",
                                                                       "km_rand_index", "precision_at_5"]}
                                                | {"total_s": "{:.1f}", "avg_tokens": "{:.1f}"}), hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(ab, x="run", y=["vocab_raw", "vocab_tfidf"], barmode="group",
                         color_discrete_sequence=["#9AA5B1", "#2E5EAA"])
            fig.update_layout(yaxis_title="terms", xaxis_title="")
            show(style(fig, 360, "Vocabulary size (raw distinct terms vs kept TF-IDF features)"))
        with c2:
            long = ab.melt(id_vars="run", value_vars=["nb_accuracy", "nb_macro_f1", "km_purity", "km_rand_index",
                                                      "precision_at_5"], var_name="metric", value_name="value")
            fig = px.bar(long, x="run", y="value", color="metric", barmode="group", color_discrete_sequence=PALETTE)
            lo = max(0.0, float(long["value"].min()) - 0.05)
            fig.update_yaxes(range=[lo, 1.0])
            fig.update_layout(xaxis_title="")
            show(style(fig, 360, "Quality metrics per configuration"))
        tcols = [c for c in ab.columns if c.startswith("t_")]
        tl = ab.melt(id_vars="run", value_vars=tcols, var_name="stage", value_name="seconds")
        tl["stage"] = tl["stage"].str.replace("t_", "").str.replace("_s", "")
        fig = px.bar(tl, x="run", y="seconds", color="stage", color_discrete_sequence=PALETTE)
        show(style(fig, 330, "Time per stage per configuration"),
             "Bigrams and no-stop-word runs are slower (bigger vocabularies); lemmatisation is slower than stemming.")
        st.download_button("Download ablation results (CSV)", ab.to_csv(index=False).encode(), "ablation_results.csv",
                           "text/csv", key="dl_ablation")
    else:
        st.info("Press the button to run the ablation study.")

    section("K-Means seed stability", "Five single-restart runs with different seeds on the current working set.")
    if st.button("Run seed-stability test (5 seeds)", key="run_seeds"):
        with st.spinner("Clustering with 5 seeds ..."):
            st.session_state.seed_stability = _seed_stability(p)
    if "seed_stability" in st.session_state:
        ss_df = st.session_state.seed_stability
        kpi_row([("Purity mean ± std", f"{ss_df['purity'].mean():.3f} ± {ss_df['purity'].std(ddof=0):.3f}"),
                 ("Rand index mean ± std", f"{ss_df['rand_index'].mean():.3f} ± {ss_df['rand_index'].std(ddof=0):.3f}"),
                 ("SSE range", f"{ss_df['SSE'].min():.1f} – {ss_df['SSE'].max():.1f}")])
        c1, c2 = st.columns([1, 1.3])
        c1.dataframe(ss_df.round(4), hide_index=True)
        long = ss_df.melt(id_vars="seed", value_vars=["purity", "rand_index"], var_name="metric")
        fig = px.bar(long.assign(seed=long["seed"].astype(str)), x="seed", y="value", color="metric",
                     barmode="group", color_discrete_sequence=["#2E5EAA", "#E07A1F"])
        fig.update_yaxes(range=[0, 1.02])
        with c2:
            show(style(fig, 300, "Purity / RI per seed"),
                 "A single K-Means run can converge to a poor local optimum. The app uses n_init restarts and keeps "
                 "the run with the lowest SSE.")
        st.download_button("Download seed-stability results (CSV)", ss_df.to_csv(index=False).encode(),
                           "kmeans_seed_stability.csv", "text/csv", key="dl_seeds")
    if st.session_state.get("ablation") is None and "seed_stability" not in st.session_state:
        st.caption("Results from both studies are also used on the Inferences page.")
