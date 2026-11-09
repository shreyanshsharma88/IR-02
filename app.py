"""News Intelligence — Text Mining & Document Intelligence System (Streamlit front end).

Run:  streamlit run app.py

All IR / ML algorithms (inverted index, TF-IDF, cosine, Multinomial Naive Bayes,
K-Means, Purity, Rand Index, recommender) live in the `textmining` package and are
implemented from scratch — scikit-learn is not used anywhere.
"""
import streamlit as st

st.set_page_config(page_title="News Intelligence — Text Mining System", page_icon="📰", layout="wide",
                   initial_sidebar_state="expanded")

from ui import (page_dashboard, page_inferences, page_ingestion, page_mining,  # noqa: E402
                page_performance, page_preprocess, page_recommend)
from ui.common import CSS  # noqa: E402
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
NAV_KEY = "nav"


def sidebar(p):
    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>📰 News Intelligence</div>"
                    "<div class='sidebar-tag'>Text Mining &amp; Document Intelligence System</div>",
                    unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), key=NAV_KEY, label_visibility="collapsed")
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
