"""Shared UI helpers: page header, consistent Plotly styling, colour maps."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PALETTE = ["#2E5EAA", "#E07A1F", "#3A9D5D", "#C8423B", "#7B5EA7", "#2B9EB3", "#B8860B", "#8C564B",
           "#D45087", "#5F6B7A"]
BEFORE_COLOR = "#9AA5B1"
AFTER_COLOR = "#2E5EAA"
SEQ_SCALE = "Blues"


def cat_colors(categories) -> dict:
    cats = sorted({str(c) for c in categories})
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cats)}


def header(title: str, subtitle: str = ""):
    st.markdown(f"<h1 class='page-title'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='page-sub'>{subtitle}</p>", unsafe_allow_html=True)


def section(title: str, help_text: str = ""):
    st.markdown(f"<h3 class='section'>{title}</h3>", unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def style(fig: go.Figure, height: int = 380, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      font=dict(size=12), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""))
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=14)))
    return fig


def show(fig: go.Figure, caption: str | None = None, key: str | None = None):
    st.plotly_chart(fig, key=key, config={"displaylogo": False})
    if caption:
        st.caption(caption)


def bar(df, x, y, color=None, title=None, height=360, orientation="v", color_map=None, text=None, **kw):
    fig = px.bar(df, x=x, y=y, color=color, orientation=orientation, text=text,
                 color_discrete_map=color_map or {}, color_discrete_sequence=PALETTE, **kw)
    return style(fig, height, title, legend=color is not None)


def heatmap(z, x, y, title=None, height=420, colorscale=SEQ_SCALE, fmt="d", xlab="", ylab=""):
    text = [[(f"{v:{fmt}}" if fmt else str(v)) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, text=text, texttemplate="%{text}",
                               hovertemplate=f"{ylab}: %{{y}}<br>{xlab}: %{{x}}<br>value: %{{z}}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=ylab)
    fig.update_xaxes(title=xlab, side="bottom")
    return style(fig, height, title, legend=False)


def kpi_row(items, border=True):
    cols = st.columns(len(items))
    for c, item in zip(cols, items):
        label, value = item[0], item[1]
        help_ = item[2] if len(item) > 2 else None
        c.metric(label, value, help=help_, border=border)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def truncate(s: str, n: int = 60) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


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
