"""Streamlit entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="News Intelligence — Text Mining System", page_icon="📰", layout="wide",
                   initial_sidebar_state="expanded")

from ui import (page_dashboard, page_inferences, page_ingestion, page_mining,  # noqa: E402
                page_performance, page_preprocess, page_recommend)
from ui.state import get_pipeline  # noqa: E402

PAGES = {
    "Dashboard": page_dashboard,
    "Data Ingestion": page_ingestion,
    "Preprocessing & Features": page_preprocess,
    "Text Mining Hub": page_mining,
    "Recommendation": page_recommend,
    "Performance Analytics": page_performance,
    "Inferences & Discussion": page_inferences,
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
h1.page-title {font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem; color: var(--text-color);}
p.page-sub {color: #6b7280; font-size: 1.0rem; margin-top: 0; margin-bottom: 1.2rem;}
h3.section {font-size: 1.15rem; font-weight: 650; margin-top: 1.6rem; margin-bottom: 0.2rem;
            border-left: 4px solid #2E5EAA; padding-left: 0.55rem;}
div[data-testid="stMetric"] {background: rgba(46,94,170,0.04);}
section[data-testid="stSidebar"] .sidebar-brand {font-weight: 700; font-size: 1.05rem; margin-bottom: 0.2rem;}
section[data-testid="stSidebar"] .sidebar-tag {color: #6b7280; font-size: 0.82rem; margin-bottom: 0.8rem;}
.status-ok {color: #3A9D5D; font-weight: 600;}
.status-off {color: #9AA5B1; font-weight: 600;}
.context-box {background: rgba(46,94,170,0.06); border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;}
.doc-card {background: rgba(120,120,120,0.07); border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.92rem;}
</style>
"""


def sidebar(p):
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>📰 News Intelligence</div>"
                    "<div class='sidebar-tag'>Text Mining &amp; Document Intelligence System</div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("**Current session**")
        st.caption(f"Dataset: {p.ds['name']}  \n"
                   f"Docs: {len(p.docs):,} · working set: {len(p.work_idx):,}"
                   f"{' (dedup)' if p.dedup else ''}  \n"
                   f"Preprocessing: `{p.cfg.label()}`  \n"
                   f"NB: α={p.nb_params['alpha']}, test={p.nb_params['test_size']}  \n"
                   f"K-Means: k={p.km_params['k']}, {p.km_params['init']}, n_init={p.km_params['n_init']}")
        st.caption("Algorithms implemented from scratch — no scikit-learn.")
    return page


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    p = get_pipeline()
    page = sidebar(p)
    PAGES[page].render(p)


main()
