import os

import pandas as pd
import streamlit as st


DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATASET_PATH)
    for c in ["views", "likes", "comments"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def render() -> None:
    st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Creator Intelligence Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">A full-stack YouTube growth workstation with AI content systems, live analytics, SEO tooling, and experimental creator extensions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    df = _load_dataset()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Rows", f"{len(df):,}" if not df.empty else "0")
    c2.metric("Channels", f"{df['channel_id'].nunique():,}" if not df.empty and "channel_id" in df.columns else "0")
    c3.metric("Total Views", f"{int(df['views'].fillna(0).sum()):,}" if not df.empty and "views" in df.columns else "0")
    c4.metric("Tools", "8+ in Ytuber")

    st.markdown("### Suite Modules")
    st.markdown(
        """
<div class="tool-grid">
  <div class="tool-card"><h4>Ytuber Pro</h4><p>Live API sync, channel audit, trend radar, planner, competitor intelligence, AI studio.</p></div>
  <div class="tool-card"><h4>Recommendations Lab</h4><p>Data-driven recommendations plus thumbnail generation workflows.</p></div>
  <div class="tool-card"><h4>Channel Analysis</h4><p>KPI snapshots, publishing trends, and high-performing video diagnostics.</p></div>
  <div class="tool-card"><h4>Extension Center</h4><p>Enable/disable modules and shape your own creator OS.</p></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick Start")
    st.write("1. Open **Ytuber** and load a channel with your YouTube API key.")
    st.write("2. Run **Keyword Intel** + **Title & SEO Lab** to lock in content direction.")
    st.write("3. Use **AI Studio** for titles, descriptions, scripts, and thumbnails.")
