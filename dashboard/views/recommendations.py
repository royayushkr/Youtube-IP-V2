import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from src.llm_integration.thumbnail_generator import ThumbnailGenerator, get_api_key


if load_dotenv:
    load_dotenv()

DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")
STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "for", "with", "on", "and", "or", "at", "is", "are", "was", "were",
    "this", "that", "how", "why", "what", "when", "from", "your", "you", "my", "we", "our", "it",
}


def _load_recommendation_data() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATASET_PATH)
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["video_publishedAt"] = pd.to_datetime(df["video_publishedAt"], errors="coerce", utc=True)
    df["engagement_rate"] = ((df["likes"].fillna(0) + df["comments"].fillna(0)) / df["views"].clip(lower=1))
    df["title_length"] = df["video_title"].fillna("").astype(str).str.len()
    df["publish_day"] = df["video_publishedAt"].dt.day_name()
    return df


def _extract_keywords(titles: pd.Series, top_n: int = 8) -> list[str]:
    words: list[str] = []
    for title in titles.dropna().astype(str):
        tokens = re.findall(r"[A-Za-z]{3,}", title.lower())
        words.extend([tok for tok in tokens if tok not in STOPWORDS])
    return [w for w, _ in Counter(words).most_common(top_n)]


def _render_data_recommendations(df: pd.DataFrame) -> None:
    st.subheader("Data-Driven Content Recommendations")

    if df.empty:
        st.info("Dataset not found yet. Add `data/youtube api data/research_science_channels_videos.csv` to enable analytics recommendations.")
        return

    channels = sorted(df["channel_title"].dropna().unique().tolist())
    selected_channel = st.selectbox("Benchmark channel", ["All channels"] + channels, index=0)
    working = df if selected_channel == "All channels" else df[df["channel_title"] == selected_channel]

    if working.empty:
        st.warning("No data available for selected channel.")
        return

    high_perf_threshold = working["views"].quantile(0.75)
    high_perf = working[working["views"] >= high_perf_threshold].copy()
    if high_perf.empty:
        high_perf = working.nlargest(50, "views")

    best_day = (
        high_perf.groupby("publish_day")["views"].mean().sort_values(ascending=False).index[0]
        if high_perf["publish_day"].notna().any()
        else "N/A"
    )
    recommended_title_len = int(high_perf["title_length"].median()) if high_perf["title_length"].notna().any() else 60
    top_keywords = _extract_keywords(high_perf["video_title"], top_n=10)

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Publish Day", best_day)
    c2.metric("Target Title Length", f"~{recommended_title_len} chars")
    c3.metric("High-Perf Sample", f"{len(high_perf):,} videos")

    st.markdown("**Suggested keyword angles:** " + (", ".join(top_keywords) if top_keywords else "No strong keywords found"))

    top_refs = high_perf[
        ["channel_title", "video_title", "views", "likes", "comments", "engagement_rate", "video_publishedAt"]
    ].sort_values("views", ascending=False).head(12)
    st.markdown("**Reference videos to model:**")
    st.dataframe(top_refs, use_container_width=True)


def render() -> None:
    st.title("Recommendations & Thumbnail Generator")
    st.write("Use analytics-backed recommendations, then generate thumbnail concepts.")

    rec_df = _load_recommendation_data()
    _render_data_recommendations(rec_df)
    st.markdown("---")

    st.subheader("Thumbnail Generator")

    col1, col2 = st.columns(2)
    with col1:
        provider = st.selectbox("Provider", ["gemini", "openai"], index=0)
    with col2:
        if provider == "gemini":
            model = st.text_input(
                "Gemini image model",
                value="gemini-2.0-flash-preview-image-generation",
            )
        else:
            model = st.text_input("OpenAI image model", value="gpt-image-1")

    api_key_default = get_api_key(provider) or ""
    api_key = st.text_input(
        "API key",
        value=api_key_default,
        type="password",
        help="If blank, app reads from .env.",
    )

    title = st.text_input("Video title", value="The Physics of Black Holes in 10 Minutes")
    context = st.text_area(
        "Context",
        value=(
            "Audience: curious high-school and college students. "
            "Goal: simplify Hawking radiation and event horizon visuals."
        ),
        height=120,
    )
    style = st.text_area(
        "Style",
        value="Bold contrast, cinematic lighting, one main object, science aesthetic.",
        height=90,
    )
    negative_prompt = st.text_input(
        "Avoid",
        value="clutter, tiny text, low contrast, too many subjects",
    )

    col3, col4 = st.columns(2)
    with col3:
        count = st.slider("Number of options", min_value=1, max_value=4, value=2)
    with col4:
        size = st.selectbox(
            "Output size (OpenAI only)",
            ["1024x1024", "1536x1024", "1024x1536"],
            index=1,
        )

    run = st.button("Generate Thumbnails", type="primary", use_container_width=True)
    if not run:
        return

    if not api_key:
        st.error("Missing API key. Add it in the API key box or .env.")
        return
    if not title.strip() or not context.strip():
        st.error("Title and context are required.")
        return

    with st.spinner("Generating thumbnail concepts..."):
        try:
            generator = ThumbnailGenerator(provider=provider, api_key=api_key, model=model)
            images = generator.generate(
                title=title,
                context=context,
                style=style,
                negative_prompt=negative_prompt,
                count=count,
                size=size,
            )
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

    st.success(f"Generated {len(images)} image(s).")
    out_dir = os.path.join("outputs", "thumbnails")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for idx, generated in enumerate(images, start=1):
        st.image(generated.image_bytes, caption=f"Option {idx} ({generated.model})")
        ext = "png" if "png" in generated.mime_type else "jpg"
        filename = f"thumbnail_{ts}_{idx}.{ext}"
        file_path = os.path.join(out_dir, filename)
        with open(file_path, "wb") as fp:
            fp.write(generated.image_bytes)
        st.download_button(
            label=f"Download option {idx}",
            data=generated.image_bytes,
            file_name=filename,
            mime=generated.mime_type,
            use_container_width=True,
            key=f"download_{idx}_{ts}",
        )

    st.caption(f"Saved generated files to `{out_dir}`")
