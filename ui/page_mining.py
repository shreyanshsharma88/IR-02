"""Text mining hub: Naive Bayes, K-Means, evaluation, analytics."""
from __future__ import annotations

import random
from math import comb

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from textmining import metrics as M
from textmining.kmeans import elbow
from textmining.preprocessing import preprocess

from .common import PALETTE, cat_colors, header, heatmap, kpi_row, section, show, style
from .state import Pipeline


def _nb_tab(p: Pipeline):
    prm = p.nb_params
    with st.form("nb_form"):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        test_size = c1.slider("Test size", 0.1, 0.5, float(prm["test_size"]), 0.05)
        alpha = c2.number_input("Laplace α", 0.01, 10.0, float(prm["alpha"]), 0.1, format="%.2f")
        seed = c3.number_input("Split seed", 0, 10_000, int(prm["seed"]))
        c4.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        if c4.form_submit_button("Train Naive Bayes", type="primary"):
            st.session_state.nb_params = dict(test_size=float(test_size), alpha=float(alpha), seed=int(seed))
            st.rerun()
    st.latex(r"\hat c = \arg\max_c \Big[\log P(c) + \sum_{t\in d} tf_{t,d}\,\log\frac{T_{ct}+\alpha}"
             r"{\sum_{t'}T_{ct'}+\alpha|V|}\Big]")
    if p.n_classes < 2:
        st.warning("The loaded dataset has fewer than two categories — classification needs labelled data.")
        return
    nb = p.nb
    rep, model = nb["report"], nb["model"]
    kpi_row([("Accuracy", f"{rep['accuracy']:.4f}"), ("Macro F1", f"{rep['macro']['f1']:.4f}"),
             ("Weighted F1", f"{rep['weighted']['f1']:.4f}"),
             ("Train / test", f"{len(nb['train_idx'])}/{len(nb['test_idx'])}", "documents in the stratified split"),
             ("Vocabulary", f"{len(model.vocab):,}", "built from the training split only (no leakage)"),
             ("Leaked test docs", f"{nb['leakage']}", "test documents whose exact text also appears in the training "
                                                      "set (duplicates) — inflate accuracy")])
    if nb["leakage"] and not p.dedup:
        st.caption(f"⚠ {nb['leakage']} test articles have an exact copy in the training split. Enable "
                   "'Remove duplicates before modelling' on the Preprocessing page to measure leakage-free accuracy.")

    section("Predictions on the test set")
    te = nb["test_idx"]
    pred_df = pd.DataFrame({"doc_id": p.meta_w["doc_id"].values[te], "title": p.meta_w["title"].values[te],
                            "actual": nb["y_true"], "predicted": nb["y_pred"]})
    pred_df["correct"] = pred_df["actual"] == pred_df["predicted"]
    only_wrong = st.checkbox(f"Show misclassified only ({int((~pred_df['correct']).sum())})", key="nb_wrong")
    st.dataframe(pred_df[~pred_df["correct"]] if only_wrong else pred_df, hide_index=True, height=280)

    section("Metrics")
    rows = [{"class": c, "precision": v["precision"], "recall": v["recall"], "f1": v["f1"], "support": v["support"]}
            for c, v in rep["per_class"].items()]
    n = len(nb["y_true"])
    rows.append({"class": "macro avg", **rep["macro"], "support": n})
    rows.append({"class": "weighted avg", **rep["weighted"], "support": n})
    mt = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(mt.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), hide_index=True)
        st.caption(f"Overall accuracy = {rep['accuracy']:.4f}. Precision = TP/(TP+FP), recall = TP/(TP+FN), "
                   "F1 = harmonic mean; all computed from the hand-built confusion matrix.")
    with c2:
        labels = rep["labels"]
        dist = pd.DataFrame({"class": labels * 2,
                             "documents": [nb["y_true"].count(c) for c in labels] + [nb["y_pred"].count(c) for c in labels],
                             "series": ["actual"] * len(labels) + ["predicted"] * len(labels)})
        fig = px.bar(dist, x="class", y="documents", color="series", barmode="group", text="documents",
                     color_discrete_map={"actual": "#9AA5B1", "predicted": "#2E5EAA"})
        show(style(fig, 330, "Actual vs predicted class distribution"),
             "Bars of equal height per class mean the classifier neither over- nor under-predicts that desk.")

    c1, c2 = st.columns([1, 1])
    with c1:
        cm = rep["confusion_matrix"]
        fig = heatmap(cm, rep["labels"], rep["labels"], "Confusion matrix (rows = actual, cols = predicted)",
                      height=430, xlab="predicted", ylab="actual")
        show(fig, "Off-diagonal cells are errors; typical confusions are business ↔ politics and business ↔ tech.")
    with c2:
        st.markdown("**Top indicative terms per class**")
        st.caption("Ranked by log P(t|c) − mean log P(t|c') (class-discriminative log-likelihood ratio).")
        feats = model.top_features(12)
        st.dataframe(pd.DataFrame({c: [t for t, _ in v] + [""] * (12 - len(v)) for c, v in feats.items()}),
                     hide_index=True, height=400)

    section("Classify your own text")
    txt = st.text_area("Paste a headline or article", "The central bank raised interest rates as inflation and "
                       "share prices climbed, worrying investors and markets.", key="nb_text", height=90)
    toks = preprocess(txt, p.cfg)
    if toks:
        proba = model.predict_proba_one(toks)
        pr = pd.DataFrame(sorted(proba.items(), key=lambda x: -x[1]), columns=["class", "posterior"])
        known = sum(t in model.vocab for t in toks)
        c1, c2 = st.columns([1, 2])
        c1.metric("Predicted desk", pr["class"].iat[0], f"P = {pr['posterior'].iat[0]:.3f}", delta_color="off")
        c1.caption(f"{len(toks)} tokens after preprocessing, {known} in the training vocabulary: "
                   f"`{' '.join(toks[:25])}`")
        fig = px.bar(pr, x="posterior", y="class", orientation="h", color="class",
                     color_discrete_map=cat_colors(pr["class"]), text=pr["posterior"].map(lambda v: f"{v:.3f}"))
        fig.update_yaxes(autorange="reversed", title="")
        fig.update_xaxes(range=[0, 1])
        with c2:
            show(style(fig, 260, "Posterior probabilities (log-sum-exp normalised)", legend=False))
    else:
        st.info("No tokens left after preprocessing.")


def _km_tab(p: Pipeline):
    prm = p.km_params
    with st.form("km_form"):
        c = st.columns(6)
        k = c[0].number_input("k (clusters)", 2, 20, int(prm["k"]))
        init = c[1].selectbox("Initialisation", ["kmeans++", "random"], index=["kmeans++", "random"].index(prm["init"]))
        max_iter = c[2].number_input("max_iter", 1, 500, int(prm["max_iter"]))
        tol = c[3].number_input("tol (centroid shift)", 0.0, 1.0, float(prm["tol"]), format="%.5f", step=1e-4)
        n_init = c[4].number_input("n_init (restarts)", 1, 20, int(prm["n_init"]))
        seed = c[5].number_input("seed", 0, 10_000, int(prm["seed"]))
        if st.form_submit_button("Run K-Means", type="primary"):
            st.session_state.km_params = dict(k=int(k), init=init, max_iter=int(max_iter), tol=float(tol),
                                              n_init=int(n_init), seed=int(seed))
            st.session_state.pop("elbow", None)
            st.rerun()
    km, ev = p.km, p.km_eval
    meta = p.meta_w
    kpi_row([("Iterations (best run)", f"{km.n_iter_}"), ("SSE (best run)", f"{km.sse_:.2f}"),
             ("Purity", f"{ev['purity']:.4f}"), ("Rand index", f"{ev['rand_index']:.4f}"),
             ("Documents clustered", f"{len(km.labels_):,}"), ("Features", f"{p.X.shape[1]:,}")])
    st.info(f"**Stop reason:** {km.stop_reason}")

    with st.expander("Algorithm steps", expanded=False):
        st.markdown(
            "1. **Initialise** k centroids — *k-means++* picks the first seed uniformly, then each next seed with "
            "probability ∝ D(x)² (distance to nearest chosen seed); *random* picks k distinct documents.\n"
            "2. **Assign** each document to the nearest centroid (squared Euclidean on L2-normalised TF-IDF rows: "
            "‖x−c‖² = 1 − 2x·c + ‖c‖²).\n"
            "3. **Update** every centroid to the mean of its members (empty cluster → re-seeded with farthest doc).\n"
            "4. **Stop** when no document changes cluster, or the largest centroid shift < tol, or max_iter is reached.\n"
            "5. Repeat for `n_init` restarts and keep the run with the lowest SSE.")

    section("Step 1 · Initialisation (seed documents of the best run)")
    init_df = pd.DataFrame({"centroid": range(len(km.init_indices)),
                            "doc_id": meta["doc_id"].values[km.init_indices],
                            "title": meta["title"].values[km.init_indices],
                            "true category": meta["category"].values[km.init_indices]})
    st.dataframe(init_df, hide_index=True)

    section("Steps 2–4 · Assignment / update iterations")
    hist = pd.DataFrame(km.history)
    hist_view = hist.copy()
    hist_view["cluster_sizes"] = hist_view["cluster_sizes"].map(lambda s: " / ".join(map(str, s)))
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(hist_view, hide_index=True, height=330)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["iteration"], y=hist["SSE"], name="SSE", mode="lines+markers",
                                 line=dict(color="#2E5EAA")))
        fig.add_trace(go.Bar(x=hist["iteration"], y=hist["docs_changed"], name="docs changed", yaxis="y2",
                             marker_color="rgba(224,122,31,0.55)"))
        fig.update_layout(yaxis=dict(title="SSE"), xaxis=dict(title="iteration"),
                          yaxis2=dict(title="documents re-assigned", overlaying="y", side="right", showgrid=False))
        show(style(fig, 330, "Convergence: SSE and re-assignments per iteration"),
             "SSE decreases monotonically; re-assignments fall to zero (or the centroid shift drops below tol).")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(hist, x="iteration", y="max_centroid_shift", markers=True, log_y=True,
                      color_discrete_sequence=["#3A9D5D"])
        fig.add_hline(y=max(km.tol, 1e-12), line_dash="dot", annotation_text="tol")
        show(style(fig, 280, "Largest centroid shift per iteration (log)", legend=False))
    with c2:
        rs = pd.DataFrame(km.run_summary)
        rs["kept"] = rs["SSE"] == rs["SSE"].min()
        fig = px.bar(rs, x="run", y="SSE", color="kept", text="iterations",
                     color_discrete_map={True: "#2E5EAA", False: "#9AA5B1"})
        fig.update_yaxes(range=[rs["SSE"].min() * 0.98, rs["SSE"].max() * 1.01])
        show(style(fig, 280, "Restarts: final SSE (label = iterations)"),
             "Different initial seeds converge to different local optima; the lowest-SSE run is kept.")

    section("Clusters")
    cl = km.labels_
    names = {j: f"C{j} ({ev['majority'].get(j, '-')})" for j in range(km.k)}
    color_by = st.radio("Colour scatter by", ["cluster", "true category"], horizontal=True, key="km_color")
    pts = p.pca
    sc = pd.DataFrame({"PC1": pts[:, 0], "PC2": pts[:, 1], "cluster": [names[c] for c in cl],
                       "true category": p.labels, "title": meta["title"].values, "doc_id": meta["doc_id"].values})
    cmap = ({n: PALETTE[j % len(PALETTE)] for j, n in names.items()} if color_by == "cluster"
            else cat_colors(sc["true category"]))
    sc_sorted = sc.sort_values(color_by)
    fig = px.scatter(sc_sorted, x="PC1", y="PC2", color=color_by, hover_data=["doc_id", "title", "cluster",
                                                                                 "true category"],
                     color_discrete_map=cmap, opacity=0.75, render_mode="webgl")
    fig.update_traces(marker=dict(size=5))
    c1, c2 = st.columns([1.6, 1])
    with c1:
        show(style(fig, 470, "2-D PCA projection of TF-IDF vectors"),
             "PCA (numpy eigen-decomposition) is used for visualisation only — clustering runs in the full "
             f"{p.X.shape[1]:,}-dimensional TF-IDF space, so overlap in 2-D does not imply overlap in the model.")
    with c2:
        sizes = pd.DataFrame({"cluster": [names[j] for j in range(km.k)],
                              "documents": np.bincount(cl, minlength=km.k)})
        fig = px.bar(sizes, x="cluster", y="documents", text="documents", color="cluster",
                     color_discrete_map={n: PALETTE[j % len(PALETTE)] for j, n in names.items()})
        show(style(fig, 470, "Cluster sizes", legend=False), "Cluster label = majority true category.")

    c1, c2 = st.columns([1, 1])
    with c1:
        tt = km.top_terms(p.tfidf.vocab, 10)
        st.markdown("**Top terms per cluster (highest centroid weights)**")
        st.dataframe(pd.DataFrame({"cluster": [names[j] for j in tt], "size": np.bincount(cl, minlength=km.k),
                                   "top terms": [", ".join(v) for v in tt.values()]}), hide_index=True)
    with c2:
        st.markdown("**Cluster membership**")
        sel = st.selectbox("Filter cluster", ["all"] + [names[j] for j in range(km.k)], key="km_filter")
        mem = sc[["doc_id", "title", "true category", "cluster"]]
        st.dataframe(mem if sel == "all" else mem[mem["cluster"] == sel], hide_index=True, height=300)

    section("Elbow analysis (optional)")
    if st.button("Compute elbow curve for k = 2 … 10", key="km_elbow_btn"):
        with st.spinner("Running K-Means for k = 2..10 ..."):
            st.session_state.elbow = elbow(p.X, range(2, 11), seed=int(prm["seed"]), init=prm["init"])
    if "elbow" in st.session_state:
        eb = pd.DataFrame(st.session_state.elbow, columns=["k", "SSE"])
        fig = px.line(eb, x="k", y="SSE", markers=True, color_discrete_sequence=["#2E5EAA"])
        fig.add_vline(x=km.k, line_dash="dot", annotation_text=f"current k={km.k}")
        show(style(fig, 320, "SSE vs k", legend=False),
             "SSE always falls as k grows; look for the bend where extra clusters stop paying off.")


def interpretation(p: Pipeline) -> str:
    rep, ev = p.nb["report"], p.km_eval
    acc, f1 = rep["accuracy"], rep["macro"]["f1"]
    pur, ri = ev["purity"], ev["rand_index"]
    k = p.km.k
    worst = min(rep["per_class"].items(), key=lambda kv: kv[1]["f1"])
    gap = acc - pur
    lines = [
        f"**Naive Bayes** (supervised) reaches accuracy **{acc:.3f}** and macro-F1 **{f1:.3f}** on "
        f"{len(p.nb['test_idx'])} held-out articles. It learns P(term | desk) directly from labelled examples, so its "
        f"decision boundary is aligned with the editorial categories. Its weakest class is **{worst[0]}** "
        f"(F1 {worst[1]['f1']:.3f}), whose vocabulary overlaps most with other desks.",
        f"**K-Means** (unsupervised, k={k}) never sees a label, yet its clusters reach purity **{pur:.3f}** and Rand "
        f"index **{ri:.3f}**. That the clusters recover the desks this well shows the five categories form genuinely "
        f"separable regions of the TF-IDF space.",
        ("The supervised model is " + (f"**{gap * 100:.1f} points higher**" if gap > 0 else
                                        f"**{-gap * 100:.1f} points lower**") +
         " (accuracy vs purity). The numbers are **not strictly comparable**: accuracy is measured on a held-out "
         "test split, while purity / RI are computed on the whole corpus after mapping each cluster to its majority "
         "class. Purity rewards many small clusters (k = N gives purity 1) and RI is inflated by the large number of "
         "true-negative pairs "
         f"(TN = {ev['counts']['TN']:,} of {ev['counts']['pairs']:,} pairs)."),
        "**Business reading:** use Naive Bayes to route incoming articles to known desks (labels exist, high "
        "accuracy, explainable via top terms); use K-Means to discover emerging themes and sub-topics that no desk "
        "label captures yet, then promote stable clusters to new labelled categories.",
    ]
    if p.nb["leakage"] and not p.dedup:
        lines.append(f"Note: {p.nb['leakage']} test articles have exact copies in training, which slightly inflates "
                     "accuracy — enable de-duplication on the Preprocessing page for a leakage-free estimate.")
    return "\n\n".join(lines)


def _eval_tab(p: Pipeline):
    km, ev = p.km, p.km_eval
    cnt = ev["counts"]
    st.latex(r"\text{Purity}(\Omega,C)=\frac1N\sum_k\max_j|\omega_k\cap c_j| \qquad "
             r"RI=\frac{TP+TN}{TP+FP+FN+TN}=\frac{TP+TN}{\binom N2}")
    kpi_row([("Purity", f"{ev['purity']:.4f}"), ("Rand index", f"{ev['rand_index']:.4f}"),
             ("TP", f"{cnt['TP']:,}", "same cluster & same class"), ("FP", f"{cnt['FP']:,}", "same cluster, different class"),
             ("FN", f"{cnt['FN']:,}", "different cluster, same class"),
             ("TN", f"{cnt['TN']:,}", "different cluster & different class"),
             ("C(N,2) pairs", f"{cnt['pairs']:,}")])
    st.caption(f"Pair counts from the contingency table: TP = Σ C(n_ij,2), TP+FP = Σ_k C(|ω_k|,2), "
               f"TP+FN = Σ_j C(|c_j|,2). Check: TP+FP+FN+TN = {cnt['TP'] + cnt['FP'] + cnt['FN'] + cnt['TN']:,} "
               f"= C({len(p.labels)},2) = {comb(len(p.labels), 2):,}.")

    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("Brute-force verification (300-doc sample)")
        rng = random.Random(0)
        n = len(p.labels)
        sample = sorted(rng.sample(range(n), min(300, n)))
        cs = [int(km.labels_[i]) for i in sample]
        ys = [p.labels[i] for i in sample]
        fast = M.rand_index_counts(cs, ys)
        brute = M.rand_index_bruteforce(cs, ys)
        st.dataframe(pd.DataFrame([
            {"method": "contingency-table formula", "Rand index": f"{fast['rand_index']:.10f}",
             "pairs": f"{fast['pairs']:,}"},
            {"method": "O(N²) brute-force pair loop", "Rand index": f"{brute:.10f}", "pairs": f"{comb(len(sample), 2):,}"},
        ]), hide_index=True)
        ok = abs(fast["rand_index"] - brute) < 1e-12
        (st.success if ok else st.error)(f"Both implementations {'agree' if ok else 'DISAGREE'} "
                                         f"(|Δ| = {abs(fast['rand_index'] - brute):.2e}). Sample purity = "
                                         f"{M.purity(cs, ys):.4f}.")
    with c2:
        section("Contingency table: cluster × true class")
        table = M.contingency(km.labels_.tolist(), p.labels)
        classes = sorted(set(p.labels))
        clusters = sorted(table)
        z = [[table[c].get(y, 0) for y in classes] for c in clusters]
        ylabels = [f"C{c} → {ev['majority'][c]}" for c in clusters]
        fig = heatmap(z, classes, ylabels, height=360, xlab="true class", ylab="cluster (→ majority label)")
        show(fig, "Each row's maximum is the cluster's majority class; purity = sum of row maxima / N.")

    section("Supervised vs unsupervised comparison")
    if p.n_classes < 2:
        st.warning("Need labelled data with ≥ 2 classes for the comparison.")
        return
    rep = p.nb["report"]
    comp = pd.DataFrame([
        {"model": "Naive Bayes (supervised)", "metric": "Accuracy", "value": rep["accuracy"]},
        {"model": "Naive Bayes (supervised)", "metric": "Macro precision", "value": rep["macro"]["precision"]},
        {"model": "Naive Bayes (supervised)", "metric": "Macro recall", "value": rep["macro"]["recall"]},
        {"model": "Naive Bayes (supervised)", "metric": "Macro F1", "value": rep["macro"]["f1"]},
        {"model": "K-Means (unsupervised)", "metric": "Purity", "value": ev["purity"]},
        {"model": "K-Means (unsupervised)", "metric": "Rand index", "value": ev["rand_index"]},
    ])
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.dataframe(comp.assign(value=comp["value"].round(4)), hide_index=True)
    with c2:
        fig = px.bar(comp, x="metric", y="value", color="model", text=comp["value"].map(lambda v: f"{v:.3f}"),
                     color_discrete_sequence=["#2E5EAA", "#E07A1F"])
        fig.update_yaxes(range=[0, 1.05])
        show(style(fig, 330, "Naive Bayes vs K-Means"))
    section("Interpretation (auto-generated from the current run)")
    st.markdown(interpretation(p))


@st.cache_resource(show_spinner=False, max_entries=6)
def _cat_terms(key, cfg_json, tfidf_json, subset_key, _X, _labels, _vocab, n=15):
    labels = np.array(_labels)
    out = {}
    for c in sorted(set(_labels)):
        mean = _X[labels == c].mean(axis=0)
        idx = np.argsort(-mean)[:n]
        out[c] = [(_vocab[i], float(mean[i])) for i in idx]
    return out


def _analytics_tab(p: Pipeline):
    section("Top TF-IDF terms per category", "Mean TF-IDF weight of each term across the category's articles.")
    ct = _cat_terms(p.key, p.cfg_json, p.tfidf_json, p.subset_key, p.X, p.labels, p.tfidf.vocab)
    cats = list(ct)
    cmap = cat_colors(cats)
    for row_start in range(0, len(cats), 3):
        cols = st.columns(3)
        for col, c in zip(cols, cats[row_start:row_start + 3]):
            d = pd.DataFrame(ct[c], columns=["term", "mean tf-idf"])
            fig = px.bar(d, x="mean tf-idf", y="term", orientation="h", color_discrete_sequence=[cmap[c]])
            fig.update_yaxes(autorange="reversed", title="")
            with col:
                show(style(fig, 380, c, legend=False))

    section("Keyword search (Boolean AND over the inverted index)")
    q = st.text_input("Query terms", "oil price", key="kw_query",
                      help="Each word is preprocessed with the current pipeline; documents must contain ALL terms.")
    terms = []
    for w in q.split():
        terms.extend(preprocess(w, p.cfg)[:1])
    terms = list(dict.fromkeys(terms))
    if not terms:
        st.info("Enter at least one content word.")
        return
    idx = p.tfidf.index
    postings = [set(idx.lookup(t)) for t in terms]
    hits = set.intersection(*postings) if postings else set()
    st.caption("Query terms after preprocessing: " + ", ".join(f"`{t}` (df={len(s)})" for t, s in zip(terms, postings))
               + f" → **{len(hits)} documents** match all terms.")
    if hits:
        cols = [p.tfidf.term_to_col[t] for t in terms if t in p.tfidf.term_to_col]
        rows = []
        for d in hits:
            score = float(p.X[d, cols].sum()) if cols else 0.0
            rows.append({"doc_id": p.meta_w["doc_id"].iat[d], "title": p.meta_w["title"].iat[d],
                         "category": p.meta_w["category"].iat[d], "score (Σ tf-idf of query terms)": round(score, 4),
                         **{f"tf({t})": idx.lookup(t).get(d, 0) for t in terms}})
        res = pd.DataFrame(rows).sort_values("score (Σ tf-idf of query terms)", ascending=False)
        c1, c2 = st.columns([2, 1])
        c1.dataframe(res, hide_index=True, height=320)
        cc = res["category"].value_counts().rename_axis("category").reset_index(name="hits")
        fig = px.pie(cc, names="category", values="hits", color="category", color_discrete_map=cat_colors(cc["category"]),
                     hole=0.45)
        with c2:
            show(style(fig, 320, "Matches by category"))


def render(p: Pipeline):
    header("Text Mining Hub", "Supervised classification, unsupervised clustering, evaluation and corpus analytics "
                              "— all algorithms implemented from scratch.")
    st.caption(f"Working set: {len(p.work_idx):,} documents ({'deduplicated' if p.dedup else 'all documents'}) · "
               f"preprocessing `{p.cfg.label()}` · {p.X.shape[1]:,} TF-IDF features")
    tabs = st.tabs(["Naive Bayes classification", "K-Means clustering", "Evaluation & comparison",
                    "Corpus analytics"])
    with tabs[0]:
        _nb_tab(p)
    with tabs[1]:
        _km_tab(p)
    with tabs[2]:
        _eval_tab(p)
    with tabs[3]:
        _analytics_tab(p)
