"""Shared theme: colour palette, fonts and global CSS injection.

Keeping the palette in one place means the custom-HTML components and the
native Streamlit theme stay in sync.
"""

from __future__ import annotations

import streamlit as st

# ── Core palette ──────────────────────────────────────────────────────────
C = {
    "bg": "#f5f3ee",
    "card": "#ffffff",
    "primary": "#2d6a4f",
    "primaryLight": "#40916c",
    "primaryPale": "#d8f3dc",
    "accent": "#e76f51",
    "accentPale": "#fae1db",
    "text": "#1b1b1b",
    "textMuted": "#6b705c",
    "border": "#e0ddd5",
    "tagBg": "#edf2f4",
    # Nutri-Score (Santé publique France, March 2024 charter — Appendix B)
    "nutA": "#038141",
    "nutB": "#85bb2f",
    "nutC": "#fecb02",
    "nutD": "#ee8100",
    "nutE": "#e63e11",
    "off": "#c8c5bd",
    "warn": "#b45309",
    "sub": "#7c3aed",
}

# Macronutrient accent colours used by the donut / bars
MACRO = {"protein": "#e07a5f", "carbs": "#81b29a", "fat": "#f2cc8f"}

NUTRI_C = {"A": C["nutA"], "B": C["nutB"], "C": C["nutC"], "D": C["nutD"], "E": C["nutE"]}

F = "'Source Sans 3', sans-serif"
FP = "'Playfair Display', Georgia, serif"


def inject_global_css() -> None:
    """Load web fonts and apply page-wide styling. Call once per rerun."""
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
          html, body, [class*="css"] {{ font-family: {F}; }}
          .stApp {{ background: {C['bg']}; }}
          /* Keep enough top padding to clear the fixed Streamlit toolbar, while
             staying denser than the default for an app-like feel */
          .block-container {{ padding-top: 3.5rem; padding-bottom: 3rem; max-width: 820px; }}
          /* Headings use the display serif */
          h1, h2, h3 {{ font-family: {FP}; color: {C['primary']}; }}
          /* Primary buttons */
          .stButton > button {{ font-family: {F}; font-weight: 600; }}
          /* Visible focus ring for keyboard nav (WCAG 2.4.7) */
          a:focus-visible, button:focus-visible, input:focus-visible,
          select:focus-visible, [tabindex]:focus-visible {{
            outline: 3px solid {C['primaryLight']}; outline-offset: 2px;
          }}
          .foodrag-brand {{ font-family: {FP}; color: {C['primary']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def html(markup: str) -> None:
    """Render a raw HTML fragment (inline-styled component markup)."""
    st.markdown(markup, unsafe_allow_html=True)
