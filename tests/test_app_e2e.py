"""End-to-end tests of the Streamlit app.

* streamlit.testing.v1.AppTest drives the real `app.py` script headlessly: every page,
  every major button / form / widget, dedup toggle, uploads (via the same loader the
  ingestion page uses), recommendation modes.
* One live-server smoke test launches `streamlit run app.py` and polls its health endpoint.

Streamlit's `st.cache_resource` is process-wide, so after the first test the heavy
pipeline steps are cached and the remaining tests are quick.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
PAGES = ["Dashboard", "Data Ingestion", "Preprocessing & Features", "Text Mining Hub", "Recommendation",
         "Performance Analytics", "Inferences & Discussion"]

pytestmark = pytest.mark.e2e


# ------------------------------------------------------------------ helpers
def new_app(timeout: int = 300) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert_ok(at)
    return at


def assert_ok(at: AppTest):
    excs = [e.value for e in at.exception]
    assert not excs, f"Streamlit exception(s): {excs}"
    errs = [e.value for e in at.error]
    assert not errs, f"st.error shown: {errs}"


def goto(at: AppTest, page: str) -> AppTest:
    at.radio(key="nav").set_value(page).run()
    assert_ok(at)
    return at


def page_text(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def visit_all(at: AppTest):
    for page in PAGES:
        goto(at, page)


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def metric(at: AppTest, label: str):
    return next(m for m in at.metric if m.label == label)


@pytest.fixture
def app() -> AppTest:
    return new_app()


# ------------------------------------------------------------------ navigation
def test_sidebar_navigation_options(app):
    assert list(app.radio(key="nav").options) == PAGES
    assert app.radio(key="nav").value == "Dashboard"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(app, page):
    goto(app, page)
    assert app.radio(key="nav").value == page
    assert "page-title" in page_text(app)


def test_dashboard_kpis(app):
    labels = [m.label for m in app.metric]
    assert labels, "dashboard shows no KPI metrics"
    assert "2,225" in " ".join(str(m.value) for m in app.metric) + page_text(app) + " ".join(
        str(c.value) for c in app.caption)


# ------------------------------------------------------------------ Text Mining Hub
def test_train_naive_bayes_form(app):
    goto(app, "Text Mining Hub")
    alpha = next(n for n in app.number_input if n.label == "Laplace α")
    alpha.set_value(0.5)
    button(app, "Train Naive Bayes").click().run()
    assert_ok(app)
    assert app.session_state["nb_params"]["alpha"] == 0.5
    acc = float(metric(app, "Accuracy").value)
    assert 0.9 < acc <= 1.0


def test_naive_bayes_default_metrics(app):
    goto(app, "Text Mining Hub")
    assert float(metric(app, "Accuracy").value) == pytest.approx(0.9617, abs=1e-4)
    assert metric(app, "Leaked test docs").value == "38"
    assert metric(app, "Train / test").value == "1781/444"


def test_classify_own_text(app):
    goto(app, "Text Mining Hub")
    app.text_area(key="nb_text").input("Chelsea beat Arsenal in the Premiership football match").run()
    assert_ok(app)
    assert metric(app, "Predicted desk").value == "sport"


def test_run_kmeans_form(app):
    goto(app, "Text Mining Hub")
    k = next(n for n in app.number_input if n.label == "k (clusters)")
    k.set_value(4)
    n_init = next(n for n in app.number_input if n.label == "n_init (restarts)")
    n_init.set_value(1)
    button(app, "Run K-Means").click().run()
    assert_ok(app)
    assert app.session_state["km_params"]["k"] == 4 and app.session_state["km_params"]["n_init"] == 1
    assert 0 < float(metric(app, "Purity").value) <= 1
    assert any("Stop reason" in str(i.value) for i in app.info)


def test_kmeans_scatter_color_toggle(app):
    goto(app, "Text Mining Hub")
    app.radio(key="km_color").set_value("true category").run()
    assert_ok(app)


@pytest.mark.slow
def test_elbow_button(app):
    goto(app, "Text Mining Hub")
    app.button(key="km_elbow_btn").click().run()
    assert_ok(app)
    eb = app.session_state["elbow"]
    assert [k for k, _ in eb] == list(range(2, 11))
    sse = [s for _, s in eb]
    assert sse[0] > sse[-1]


# ------------------------------------------------------------------ Preprocessing / dedup
def test_dedup_toggle_then_all_pages(app):
    goto(app, "Preprocessing & Features")
    app.checkbox(key="w_dedup").check().run()
    assert_ok(app)
    assert app.session_state["dedup"] is True
    assert any("(dedup)" in str(c.value) and "2,063" in str(c.value) for c in app.sidebar.caption)
    visit_all(app)
    goto(app, "Text Mining Hub")
    assert metric(app, "Leaked test docs").value == "0"
    assert metric(app, "Documents clustered").value == "2,063"
    # and back off again
    goto(app, "Preprocessing & Features")
    app.checkbox(key="w_dedup").uncheck().run()
    assert_ok(app)
    assert app.session_state["dedup"] is False


def test_dup_threshold_slider(app):
    goto(app, "Preprocessing & Features")
    app.slider(key="w_dup_threshold").set_value(0.95).run()
    assert_ok(app)
    assert app.session_state["dup_threshold"] == pytest.approx(0.95)


def test_apply_preprocessing_form_lemma(app):
    goto(app, "Preprocessing & Features")
    next(r for r in app.radio if r.label == "Normaliser").set_value("lemma")
    next(c for c in app.text_input if c.label.startswith("Custom stop-words")).input("bbc, people")
    button(app, "Apply & rebuild pipeline").click().run()
    assert_ok(app)
    assert app.session_state["cfg"].normalizer == "lemma"
    assert app.session_state["cfg"].custom_stopwords == ["bbc", "people"]
    for page in ("Text Mining Hub", "Recommendation", "Dashboard"):
        goto(app, page)


# ------------------------------------------------------------------ Recommendation
def test_recommend_pick_document_k_and_hide(app):
    goto(app, "Recommendation")
    assert app.radio(key="rec_mode").value == "Pick a document"
    df = app.dataframe[0].value
    assert len(df) == 5
    assert list(df["rank"]) == [1, 2, 3, 4, 5]
    assert df["score"].is_monotonic_decreasing and (df["score"] < 0.9).all()   # hide dups on by default

    app.slider(key="rec_k").set_value(10).run()
    assert_ok(app)
    assert len(app.dataframe[0].value) == 10

    app.selectbox(key="rec_doc").set_value(100).run()
    assert_ok(app)
    assert app.dataframe[0].value["doc_id"].iloc[0] != app.session_state["ds"]["meta"]["doc_id"].iat[100]

    app.checkbox(key="rec_hide").uncheck().run()
    assert_ok(app)
    app.slider(key="rec_k").set_value(3).run()
    assert len(app.dataframe[0].value) == 3


def test_recommend_hide_dups_removes_exact_copy(app):
    goto(app, "Recommendation")
    app.selectbox(key="rec_doc").set_value(6).run()       # BUS-007 has an exact duplicate (BUS-253)
    shown = app.dataframe[0].value
    assert "BUS-253" not in shown["doc_id"].tolist()
    app.checkbox(key="rec_hide").uncheck().run()
    assert_ok(app)
    shown = app.dataframe[0].value
    assert shown["doc_id"].iloc[0] == "BUS-253" and shown["score"].iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_recommend_threshold_slider(app):
    goto(app, "Recommendation")
    app.slider(key="rec_thr").set_value(0.8).run()
    assert_ok(app)
    assert (app.dataframe[0].value["score"] < 0.8).all()


def test_recommend_free_text_mode(app):
    goto(app, "Recommendation")
    app.radio(key="rec_mode").set_value("Free text").run()
    assert_ok(app)
    df = app.dataframe[0].value
    assert len(df) == 5 and (df["category"] == "tech").sum() >= 3
    app.text_area(key="rec_text").input("qwertyuiop zxcvbnm").run()
    assert_ok(app)
    assert any("No recommendations" in str(w.value) for w in app.warning)


# ------------------------------------------------------------------ Performance / Inferences
def test_seed_stability_button(app):
    goto(app, "Performance Analytics")
    app.button(key="run_seeds").click().run()
    assert_ok(app)
    ss_df = app.session_state["seed_stability"]
    assert list(ss_df["seed"]) == [1, 7, 42, 123, 2024]
    assert ((ss_df["purity"] > 0) & (ss_df["purity"] <= 1)).all()


@pytest.mark.slow
def test_ablation_button_then_inferences(app):
    goto(app, "Performance Analytics")
    app.button(key="run_ablation").click().run(timeout=900)
    assert_ok(app)
    ab = app.session_state["ablation"]
    assert len(ab) == 7
    base = ab[ab["run"] == "Baseline"].iloc[0]
    assert base["test_leakage_docs"] == 38 and base["nb_accuracy"] > 0.93
    dd = ab[ab["run"] == "Baseline+dedup"].iloc[0]
    assert dd["test_leakage_docs"] == 0 and dd["docs_used"] == 2063
    goto(app, "Inferences & Discussion")       # uses the stored ablation


@pytest.mark.slow
def test_inferences_run_ablation_button(app):
    goto(app, "Inferences & Discussion")
    app.button(key="inf_run_ablation").click().run(timeout=900)
    assert_ok(app)
    assert len(app.session_state["ablation"]) == 7


# ------------------------------------------------------------------ uploads
def _upload(app: AppTest, name: str, data: bytes):
    """The file_uploader widget cannot be driven by AppTest; inject the dataset exactly as the
    'Ingest uploaded file' button does (ui.state.load_upload -> set_dataset)."""
    sys.path.insert(0, ROOT)
    from ui.state import load_upload
    ds = load_upload(name, data)
    app.session_state["ds"] = ds
    app.session_state["dedup"] = False
    return ds


def test_uploaded_labelled_csv_all_pages(app, dataset):
    docs, meta = dataset
    sub = docs.merge(meta[["doc_id", "category", "title"]], on="doc_id").iloc[::15]   # 149 docs, 5 classes
    ds = _upload(app, "sample.csv", sub.to_csv(index=False).encode())
    app.run()
    assert_ok(app)
    assert ds["key"].startswith("upload:")
    assert any("sample.csv" in str(c.value) and f"Docs: {len(sub)}" in str(c.value) for c in app.sidebar.caption)
    visit_all(app)
    goto(app, "Data Ingestion")
    app.radio(key="ing_source").set_value("Default BBC News corpus").run()
    button(app, "Load default BBC dataset").click().run()
    assert_ok(app)
    assert app.session_state["ds"]["key"] == "bbc-default"


def test_uploaded_unlabelled_zip_all_pages(app, dataset):
    import io
    import zipfile
    docs = dataset[0].iloc[::40]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for d, t in zip(docs["doc_id"], docs["text"]):
            z.writestr(f"{d}.txt", f"Title of {d}\n{t}")
    ds = _upload(app, "flat.zip", buf.getvalue())
    app.run()
    assert_ok(app)
    assert set(ds["meta"]["category"]) == {"unlabelled"}
    assert any("flat.zip" in str(c.value) for c in app.sidebar.caption)
    visit_all(app)


def test_uploaded_json_all_pages(app, dataset):
    docs, meta = dataset
    sub = docs.merge(meta[["doc_id", "category"]], on="doc_id").iloc[::25]
    ds = _upload(app, "sample.json", sub.to_json(orient="records").encode())
    app.run()
    assert_ok(app)
    assert len(ds["docs"]) == len(sub) and ds["meta"]["category"].nunique() == 5
    visit_all(app)


def test_upload_source_radio_shows_uploader(app):
    goto(app, "Data Ingestion")
    app.radio(key="ing_source").set_value("Upload CSV / JSON / ZIP of .txt").run()
    assert_ok(app)


# ------------------------------------------------------------------ live server
def _free_port(preferred: int = 8611) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


@pytest.mark.slow
def test_live_server_health():
    port = _free_port()
    cmd = [sys.executable, "-m", "streamlit", "run", APP, "--server.headless", "true",
           "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        body, deadline = None, time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail("streamlit exited early:\n" + proc.stdout.read().decode(errors="ignore"))
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                    body = r.read().decode()
                    if body == "ok":
                        break
            except OSError:
                time.sleep(0.5)
        assert body == "ok"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
            assert r.status == 200 and b"<html" in r.read().lower()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
