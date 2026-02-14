import streamlit as st


def render_sidebar() -> str:
    st.sidebar.title("YouTube IP Dashboard")
    st.sidebar.caption("Channel analytics + recommendation workflows")
    page = st.sidebar.radio(
        "Navigate",
        ["Channel Analysis", "Recommendations", "Ytuber", "Deploy Notes"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Set `YOUTUBE_API_KEY` and `GEMINI_API_KEY` in `.env` for full Ytuber functionality."
    )
    return page
