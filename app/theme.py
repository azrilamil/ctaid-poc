"""Shared theme, layout helpers and copy for the CTAID POC Streamlit app.

The dark look is intentional: one palette, one card style, no default white
Streamlit surfaces. Every page composes these primitives so the product reads
as a single application.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.artwork import ensure_hero
from app.config import DISCLAIMER, DISCLAIMER_LONG

PRIMARY = "#5AC8FA"
ACCENT_WARM = "#FF7A59"
VIOLET = "#9B8CFF"
GREEN = "#43D9A3"
TEXT = "#E7EEF9"
MUTED = "#9FB0C9"
BG = "#080F1A"
SURFACE = "#0E1A2B"
SURFACE_ALT = "#122238"
BORDER = "#1E3250"

PAGES = [
    ("Home", "Overview of the proof of concept"),
    ("Data Exploration", "The synthetic table, statistics and correlations"),
    ("Model Training", "Fit four classifiers and compare real metrics"),
    ("Prediction", "Score a pattern from a form, a sample or a CSV"),
]

CSS = f"""
<style>
:root {{
  --ctaid-bg: {BG};
  --ctaid-surface: {SURFACE};
  --ctaid-surface-alt: {SURFACE_ALT};
  --ctaid-border: {BORDER};
  --ctaid-text: {TEXT};
  --ctaid-muted: {MUTED};
  --ctaid-primary: {PRIMARY};
  --ctaid-accent: {ACCENT_WARM};
  --ctaid-violet: {VIOLET};
  --ctaid-green: {GREEN};
}}

.stApp {{
  background:
    radial-gradient(1100px 520px at 12% -10%, rgba(90,200,250,0.16), transparent 62%),
    radial-gradient(900px 480px at 88% 4%, rgba(155,140,255,0.13), transparent 60%),
    linear-gradient(180deg, #0A1422 0%, {BG} 46%, #060C15 100%);
  color: var(--ctaid-text);
  font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif;
}}

[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 0.6rem; }}

.block-container {{ padding-top: 2.1rem; padding-bottom: 3.2rem; max-width: 1360px; }}

h1, h2, h3, h4 {{ color: var(--ctaid-text); letter-spacing: -0.01em; }}
p, li, span, label, div {{ color: var(--ctaid-text); }}
a {{ color: var(--ctaid-primary); }}

/* Sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0B1626 0%, #081120 100%);
  border-right: 1px solid var(--ctaid-border);
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a {{
  border-radius: 10px;
  color: var(--ctaid-muted);
}}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(90,200,250,0.10); color: var(--ctaid-text); }}

/* Cards -------------------------------------------------------------- */
.ctaid-card {{
  background: linear-gradient(160deg, rgba(18,34,56,0.95), rgba(11,22,38,0.95));
  border: 1px solid var(--ctaid-border);
  border-radius: 16px;
  padding: 18px 20px;
  height: 100%;
  box-shadow: 0 14px 34px rgba(0,0,0,0.30);
}}
.ctaid-card h3 {{ margin: 0 0 6px 0; font-size: 1.02rem; }}
.ctaid-card p {{ margin: 0; color: var(--ctaid-muted); font-size: 0.92rem; line-height: 1.5; }}

.ctaid-stat {{
  background: linear-gradient(160deg, rgba(18,34,56,0.95), rgba(10,20,34,0.95));
  border: 1px solid var(--ctaid-border);
  border-radius: 14px;
  padding: 14px 16px;
}}
.ctaid-stat .label {{
  color: var(--ctaid-muted);
  font-size: 0.74rem;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}}
.ctaid-stat .value {{
  color: var(--ctaid-text);
  font-size: 1.72rem;
  font-weight: 650;
  line-height: 1.25;
}}
.ctaid-stat .hint {{ color: var(--ctaid-muted); font-size: 0.8rem; }}

/* Banner ------------------------------------------------------------- */
.ctaid-notice {{
  display: flex;
  align-items: center;
  gap: 12px;
  background: rgba(255,122,89,0.10);
  border: 1px solid rgba(255,122,89,0.38);
  border-left: 4px solid var(--ctaid-accent);
  color: #FFD9CC;
  border-radius: 12px;
  padding: 10px 15px;
  font-size: 0.9rem;
  margin: 0.2rem 0 1.1rem 0;
}}
.ctaid-notice .dot {{
  width: 9px; height: 9px; border-radius: 50%;
  background: var(--ctaid-accent); box-shadow: 0 0 10px var(--ctaid-accent);
  flex: 0 0 auto;
}}

.ctaid-hero svg {{ width: 100%; height: auto; display: block; border-radius: 18px; }}

.ctaid-eyebrow {{
  color: var(--ctaid-primary);
  font-size: 0.76rem;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}}
.ctaid-title {{
  font-size: clamp(1.9rem, 3.4vw, 3.05rem);
  font-weight: 700;
  line-height: 1.12;
  margin: 0 0 0.5rem 0;
  color: var(--ctaid-text);
}}
.ctaid-title span {{ color: var(--ctaid-primary); }}
.ctaid-lede {{
  color: var(--ctaid-muted);
  font-size: 1.02rem;
  line-height: 1.62;
  max-width: 78ch;
  margin-bottom: 0.6rem;
}}

.ctaid-chip {{
  display: inline-block;
  border: 1px solid var(--ctaid-border);
  background: rgba(90,200,250,0.08);
  color: #BFE6FA;
  border-radius: 999px;
  padding: 3px 11px;
  font-size: 0.76rem;
  letter-spacing: 0.04em;
  margin: 0 6px 6px 0;
}}

.ctaid-footer {{
  margin-top: 2.6rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--ctaid-border);
  color: var(--ctaid-muted);
  font-size: 0.84rem;
  line-height: 1.6;
}}
.ctaid-footer strong {{ color: var(--ctaid-text); }}

/* Widgets ------------------------------------------------------------ */
.stButton > button, .stDownloadButton > button {{
  background: linear-gradient(180deg, #1B4B74, #14395C);
  color: #EAF6FF;
  border: 1px solid #2C6B9C;
  border-radius: 10px;
  font-weight: 600;
  padding: 0.5rem 1.15rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: linear-gradient(180deg, #23618F, #17456D);
  border-color: var(--ctaid-primary);
  color: #FFFFFF;
}}
.stButton > button:focus {{ box-shadow: 0 0 0 2px rgba(90,200,250,0.35); }}

[data-testid="stMetric"] {{
  background: linear-gradient(160deg, rgba(18,34,56,0.95), rgba(10,20,34,0.95));
  border: 1px solid var(--ctaid-border);
  border-radius: 14px;
  padding: 12px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: var(--ctaid-muted); font-size: 0.8rem; }}
[data-testid="stMetricValue"] {{ color: var(--ctaid-text); }}

[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border: 1px solid var(--ctaid-border);
  border-radius: 12px;
  overflow: hidden;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--ctaid-border); }}
.stTabs [data-baseweb="tab"] {{
  background: transparent;
  color: var(--ctaid-muted);
  border-radius: 10px 10px 0 0;
  padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
  background: rgba(90,200,250,0.10);
  color: var(--ctaid-text);
  border-bottom: 2px solid var(--ctaid-primary);
}}

[data-testid="stExpander"] {{
  border: 1px solid var(--ctaid-border);
  border-radius: 12px;
  background: rgba(14,26,43,0.75);
}}

.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--ctaid-primary); }}
.stProgress > div > div > div > div {{ background: var(--ctaid-primary); }}

[data-testid="stFileUploaderDropzone"] {{
  background: rgba(14,26,43,0.9);
  border: 1px dashed #2C4B70;
  border-radius: 12px;
}}

.stAlert {{ border-radius: 12px; }}

code {{ color: #BFE6FA; }}
</style>
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def apply_theme() -> None:
    """Inject the stylesheet once per rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


def notice() -> None:
    """The persistent demo notice, shown on every page."""
    st.markdown(
        f'<div class="ctaid-notice"><span class="dot"></span><span>{DISCLAIMER}</span></div>',
        unsafe_allow_html=True,
    )


def hero() -> None:
    """Render the original generated illustration as the hero band."""
    svg = _read(ensure_hero())
    st.markdown(f'<div class="ctaid-hero">{svg}</div>', unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, lede: str) -> None:
    st.markdown(
        f'<div class="ctaid-eyebrow">{eyebrow}</div>'
        f'<div class="ctaid-title">{title}</div>'
        f'<div class="ctaid-lede">{lede}</div>',
        unsafe_allow_html=True,
    )
    notice()


def chips(items: list[str]) -> None:
    st.markdown("".join(f'<span class="ctaid-chip">{item}</span>' for item in items), unsafe_allow_html=True)


def stat(label: str, value: str, hint: str = "") -> None:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="ctaid-stat"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="ctaid-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="ctaid-footer">'
        f"<strong>CTAID POC</strong> is a portfolio demonstration. {DISCLAIMER_LONG}<br>"
        "Designed and led by Mohd Azril Amil with AI-assisted development. "
        "Source, tests and documentation are in the public repository."
        "</div>",
        unsafe_allow_html=True,
    )


def section(title: str, caption: str = "") -> None:
    caption_html = f'<p style="color:{MUTED};font-size:0.9rem;margin:-2px 0 10px 0;">{caption}</p>' if caption else ""
    st.markdown(
        f'<h3 style="margin:1.4rem 0 0.2rem 0;font-size:1.22rem;">{title}</h3>{caption_html}',
        unsafe_allow_html=True,
    )


def truncate(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 2] + ".."
