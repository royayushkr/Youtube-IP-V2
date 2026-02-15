from typing import List

import streamlit as st


def render_sidebar(nav_items: List[str]) -> str:
    st.sidebar.title("YouTube IP Dashboard")
    st.sidebar.caption("Under experimentation by Ayush")

    page = st.sidebar.radio("Navigate", nav_items, index=0)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Keys: `YOUTUBE_API_KEY` for API stats, `GEMINI_API_KEY` for AI studio, `OPENAI_API_KEY` optional."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Creator Suite • Analytics • AI • Extensions")
    return page
