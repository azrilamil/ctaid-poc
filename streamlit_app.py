"""Streamlit entry point for the CTAID POC demo.

Run locally::

    streamlit run streamlit_app.py

The app is a single page router with a sidebar: Home, Data Exploration,
Model Training, Prediction.
"""

from __future__ import annotations

import streamlit as st

from app.config import DISCLAIMER
from app.pages_data import render as render_data
from app.pages_home import render as render_home
from app.pages_prediction import render as render_prediction
from app.pages_training import render as render_training
from app.theme import PAGES, apply_theme, footer, notice

PAGE_RENDERERS = {
    "Home": render_home,
    "Data Exploration": render_data,
    "Model Training": render_training,
    "Prediction": render_prediction,
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
            '<span style="font-size:1.5rem;">&#129728;</span>'
            '<div><div style="font-weight:700;letter-spacing:0.02em;font-size:1.02rem;">CTAID POC</div>'
            '<div style="color:#9FB0C9;font-size:0.74rem;letter-spacing:0.12em;text-transform:uppercase;">'
            "Atherosclerosis demo</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color:#9FB0C9;font-size:0.8rem;line-height:1.5;margin-bottom:1rem;">'
            "Computed tomography feature research, rebuilt as a public proof of concept."
            "</div>",
            unsafe_allow_html=True,
        )
        st.radio(
            "Navigation",
            options=[name for name, _caption in PAGES],
            key="page",
            format_func=lambda name: name,
        )
        for name, caption in PAGES:
            if name == st.session_state.get("page"):
                st.caption(caption)
        st.markdown("---")
        st.markdown(
            f'<div style="color:#FFB59C;font-size:0.79rem;line-height:1.5;">{DISCLAIMER}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color:#7F95B3;font-size:0.74rem;margin-top:0.9rem;">'
            "Designed and led by Mohd Azril Amil with AI-assisted development.</div>",
            unsafe_allow_html=True,
        )
    return st.session_state.get("page", PAGES[0][0])


def main() -> None:
    st.set_page_config(
        page_title="CTAID POC | CT atherosclerosis demo",
        page_icon="&#129728;",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    from app.state import get_dataset_meta

    get_dataset_meta()
    page = _sidebar()
    renderer = PAGE_RENDERERS.get(page, render_home)
    try:
        renderer()
    except Exception as exc:  # pragma: no cover - last resort UI guard
        notice()
        st.error(f"Something went wrong on this page: {exc}")
        st.exception(exc)
        footer()


main()
