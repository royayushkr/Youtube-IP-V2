import os

import pandas as pd
import streamlit as st


DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["video_publishedAt"] = pd.to_datetime(df["video_publishedAt"], errors="coerce", utc=True)
    df["engagement_rate"] = ((df["likes"].fillna(0) + df["comments"].fillna(0)) / df["views"].clip(lower=1))
    df["publish_month"] = df["video_publishedAt"].dt.to_period("M").astype(str)
    df["publish_day"] = df["video_publishedAt"].dt.day_name()
    return df


def render() -> None:
    st.title("Channel Analysis")
    st.write("Analytics for research-focused YouTube channels and videos.")

    if not os.path.exists(DATASET_PATH):
        st.warning(f"Dataset not found at `{DATASET_PATH}`")
        return

    df = _load_data()

    channels = sorted(df["channel_title"].dropna().unique().tolist())
    selected_channels = st.multiselect("Filter channels", channels, default=channels[:8])

    min_date = df["video_publishedAt"].min().date()
    max_date = df["video_publishedAt"].max().date()
    date_range = st.date_input("Published date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = df.copy()
    if selected_channels:
        filtered = filtered[filtered["channel_title"].isin(selected_channels)]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["video_publishedAt"].dt.date >= start_date)
            & (filtered["video_publishedAt"].dt.date <= end_date)
        ]

    if filtered.empty:
        st.warning("No data after filters. Broaden your channel/date filters.")
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Videos", f"{len(filtered):,}")
    k2.metric("Channels", f"{filtered['channel_id'].nunique():,}")
    k3.metric("Total Views", f"{int(filtered['views'].fillna(0).sum()):,}")
    k4.metric("Avg Views/Video", f"{int(filtered['views'].fillna(0).mean()):,}")
    k5.metric("Median Engagement", f"{filtered['engagement_rate'].median() * 100:.2f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Top Channels by Total Views")
        channel_summary = (
            filtered.groupby("channel_title", dropna=False)
            .agg(
                videos=("video_id", "count"),
                total_views=("views", "sum"),
                avg_views=("views", "mean"),
                engagement=("engagement_rate", "median"),
            )
            .sort_values("total_views", ascending=False)
            .head(15)
            .reset_index()
        )
        st.dataframe(channel_summary, use_container_width=True)

    with right:
        st.subheader("Monthly Upload Trend")
        trend = (
            filtered.groupby("publish_month", dropna=False)
            .agg(videos=("video_id", "count"), views=("views", "sum"))
            .reset_index()
            .sort_values("publish_month")
        )
        st.line_chart(trend.set_index("publish_month")[["videos", "views"]], height=320)

    st.subheader("Best Performing Videos")
    top_videos = filtered[
        [
            "channel_title",
            "video_title",
            "views",
            "likes",
            "comments",
            "engagement_rate",
            "video_publishedAt",
        ]
    ].sort_values("views", ascending=False)
    st.dataframe(top_videos.head(50), use_container_width=True)

    st.subheader("Publishing Day Performance")
    day_perf = (
        filtered.groupby("publish_day", dropna=False)
        .agg(videos=("video_id", "count"), avg_views=("views", "mean"), median_engagement=("engagement_rate", "median"))
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .dropna(how="all")
        .reset_index()
    )
    st.dataframe(day_perf, use_container_width=True)
