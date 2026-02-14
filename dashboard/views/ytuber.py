import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception

from src.llm_integration.thumbnail_generator import ThumbnailGenerator

DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")

DEFAULT_CATEGORY = "research_science"
THUMB_KEYS = ["default", "medium", "high", "standard", "maxres"]


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _join_list(x: Optional[List[str]]) -> str:
    if not x:
        return ""
    return "|".join([str(i) for i in x])


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_call_with_backoff(fn, max_retries: int = 7):
    delay = 1.0
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn()
        except HttpError as e:
            last_exc = e
            status = getattr(e.resp, "status", None)
            if status in (403, 429, 500, 503):
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
        except Exception as e:
            last_exc = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
    raise RuntimeError(f"API call failed after retries: {last_exc}")


def _yt_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def _resolve_channel_id(youtube, handle_or_query: str) -> str:
    q = handle_or_query.strip()
    if q.startswith("UC") and len(q) >= 20:
        return q

    req = youtube.search().list(part="snippet", q=q, type="channel", maxResults=1)
    resp = _api_call_with_backoff(req.execute)
    items = resp.get("items", [])

    if not items and q.startswith("@"):
        q2 = q[1:]
        req2 = youtube.search().list(part="snippet", q=q2, type="channel", maxResults=1)
        resp2 = _api_call_with_backoff(req2.execute)
        items = resp2.get("items", [])

    if not items:
        raise RuntimeError(f"No channel found for: {handle_or_query}")

    channel_id = _safe_get(items[0], ["snippet", "channelId"])
    if not channel_id:
        raise RuntimeError(f"Search returned item without channelId for: {handle_or_query}")
    return channel_id


def _fetch_channel_details(youtube, channel_id: str) -> Dict[str, Any]:
    req = youtube.channels().list(
        part="snippet,contentDetails,statistics,brandingSettings,status,topicDetails",
        id=channel_id,
        maxResults=1,
    )
    resp = _api_call_with_backoff(req.execute)
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"No channel details returned for channelId: {channel_id}")
    return items[0]


def _fetch_recent_video_ids(
    youtube,
    uploads_playlist_id: str,
    published_after_utc: datetime,
    max_videos: int = 500,
) -> List[str]:
    video_ids: List[str] = []
    page_token: Optional[str] = None
    stop = False

    while len(video_ids) < max_videos and not stop:
        req = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist_id,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=page_token,
        )
        resp = _api_call_with_backoff(req.execute)

        for it in resp.get("items", []):
            vid = _safe_get(it, ["contentDetails", "videoId"])
            published_at = _safe_get(it, ["snippet", "publishedAt"])
            if not vid or not published_at:
                continue
            dt = pd.to_datetime(published_at, errors="coerce", utc=True)
            if pd.isna(dt):
                continue
            if dt.to_pydatetime() < published_after_utc:
                stop = True
                break
            video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def _fetch_videos_details(youtube, video_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        if not chunk:
            continue
        req = youtube.videos().list(
            part="snippet,contentDetails,statistics,status,topicDetails",
            id=",".join(chunk),
            maxResults=50,
        )
        resp = _api_call_with_backoff(req.execute)
        out.extend(resp.get("items", []))
    return out


def _extract_thumbnails(thumbnails: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(thumbnails, dict):
        thumbnails = {}

    for k in THUMB_KEYS:
        v = thumbnails.get(k, {}) if isinstance(thumbnails.get(k, {}), dict) else {}
        out[f"thumb_{k}_url"] = v.get("url", "")
        out[f"thumb_{k}_width"] = v.get("width", "")
        out[f"thumb_{k}_height"] = v.get("height", "")
    return out


def _channel_fields(channel: Dict[str, Any], handle: str) -> Dict[str, Any]:
    snippet = channel.get("snippet", {}) or {}
    stats = channel.get("statistics", {}) or {}
    branding = _safe_get(channel, ["brandingSettings", "channel"], {}) or {}
    status = channel.get("status", {}) or {}
    topic = channel.get("topicDetails", {}) or {}

    uploads_pid = _safe_get(channel, ["contentDetails", "relatedPlaylists", "uploads"], "")

    return {
        "snapshot_utc": _iso_now(),
        "category_name": DEFAULT_CATEGORY,
        "channel_handle_used": handle,
        "channel_id": channel.get("id", ""),
        "channel_title": snippet.get("title", ""),
        "channel_description": snippet.get("description", ""),
        "channel_publishedAt": snippet.get("publishedAt", ""),
        "uploads_playlist_id": uploads_pid,
        "channel_country": branding.get("country", ""),
        "channel_keywords": branding.get("keywords", ""),
        "channel_defaultLanguage": branding.get("defaultLanguage", ""),
        "channel_madeForKids": status.get("madeForKids", ""),
        "channel_isLinked": status.get("isLinked", ""),
        "channel_subscriberCount": stats.get("subscriberCount", ""),
        "channel_viewCount": stats.get("viewCount", ""),
        "channel_videoCount": stats.get("videoCount", ""),
        "channel_topicCategories": _join_list(topic.get("topicCategories")),
        "channel_topicIds": _join_list(topic.get("topicIds")),
    }


def _video_row(video: Dict[str, Any], ch: Dict[str, Any]) -> Dict[str, Any]:
    sn = video.get("snippet", {}) or {}
    cd = video.get("contentDetails", {}) or {}
    stx = video.get("statistics", {}) or {}
    vs = video.get("status", {}) or {}
    tp = video.get("topicDetails", {}) or {}

    thumbs = sn.get("thumbnails", {}) or {}
    thumb_cols = _extract_thumbnails(thumbs)

    return {
        **ch,
        "video_id": video.get("id", ""),
        "video_title": sn.get("title", ""),
        "video_description": sn.get("description", ""),
        "video_publishedAt": sn.get("publishedAt", ""),
        "video_channelId": sn.get("channelId", ""),
        "video_categoryId": sn.get("categoryId", ""),
        "video_tags": _join_list(sn.get("tags")),
        "video_defaultLanguage": sn.get("defaultLanguage", ""),
        "video_defaultAudioLanguage": sn.get("defaultAudioLanguage", ""),
        **thumb_cols,
        "views": stx.get("viewCount", ""),
        "likes": stx.get("likeCount", ""),
        "comments": stx.get("commentCount", ""),
        "duration": cd.get("duration", ""),
        "caption": cd.get("caption", ""),
        "licensedContent": cd.get("licensedContent", ""),
        "definition": cd.get("definition", ""),
        "projection": cd.get("projection", ""),
        "madeForKids": vs.get("madeForKids", ""),
        "embeddable": vs.get("embeddable", ""),
        "video_topicCategories": _join_list(tp.get("topicCategories")),
        "video_topicIds": _join_list(tp.get("topicIds")),
    }


def _load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)


def _append_rows_to_dataset(new_rows: pd.DataFrame, existing_df: pd.DataFrame) -> None:
    if new_rows.empty:
        return

    if existing_df.empty:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        new_rows.to_csv(DATASET_PATH, index=False)
        return

    existing_cols = existing_df.columns.tolist()
    for c in existing_cols:
        if c not in new_rows.columns:
            new_rows[c] = ""
    for c in new_rows.columns:
        if c not in existing_cols:
            existing_df[c] = ""
            existing_cols.append(c)

    new_rows = new_rows[existing_cols]
    new_rows.to_csv(DATASET_PATH, mode="a", header=False, index=False)


def _ensure_numeric_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["views", "likes", "comments"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "video_publishedAt" in out.columns:
        out["video_publishedAt"] = pd.to_datetime(out["video_publishedAt"], errors="coerce", utc=True)
    out["engagement_rate"] = (out["likes"].fillna(0) + out["comments"].fillna(0)) / out["views"].clip(lower=1)
    out["publish_month"] = out["video_publishedAt"].dt.to_period("M").astype(str)
    return out


def _fetch_or_get_cached_channel(
    channel_query: str,
    youtube_api_key: str,
    force_refresh: bool,
) -> Tuple[pd.DataFrame, str, str, str]:
    existing_df = _load_dataset()
    existing_df = _ensure_numeric_and_dates(existing_df) if not existing_df.empty else existing_df

    youtube = _yt_client(youtube_api_key)
    channel_id = _resolve_channel_id(youtube, channel_query)
    cutoff = datetime.now(timezone.utc) - timedelta(days=365)

    cached = pd.DataFrame()
    if not existing_df.empty and "channel_id" in existing_df.columns:
        cached = existing_df[existing_df["channel_id"].astype(str) == str(channel_id)].copy()

    if not cached.empty and not force_refresh:
        cached_recent = cached[cached["video_publishedAt"] >= pd.Timestamp(cutoff)]
        if not cached_recent.empty:
            title = cached_recent["channel_title"].dropna().iloc[0] if "channel_title" in cached_recent.columns else channel_id
            return cached_recent, "dataset_cache", channel_id, title

    channel = _fetch_channel_details(youtube, channel_id)
    uploads_pid = _safe_get(channel, ["contentDetails", "relatedPlaylists", "uploads"], "")
    if not uploads_pid:
        raise RuntimeError("Channel uploads playlist not found.")

    video_ids = _fetch_recent_video_ids(youtube, uploads_pid, cutoff, max_videos=500)
    if not video_ids:
        if not cached.empty:
            title = cached["channel_title"].dropna().iloc[0] if "channel_title" in cached.columns else channel_id
            return cached, "dataset_cache", channel_id, title
        raise RuntimeError("No videos found in last 1 year for this channel.")

    videos = _fetch_videos_details(youtube, video_ids)
    ch = _channel_fields(channel, channel_query)
    rows = []
    for v in videos:
        vid = str(v.get("id", "")).strip()
        if not vid:
            continue
        rows.append(_video_row(v, ch))

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        raise RuntimeError("API returned no usable video rows.")

    if not existing_df.empty and "video_id" in existing_df.columns:
        existing_ids = set(existing_df["video_id"].dropna().astype(str).tolist())
        new_df = new_df[~new_df["video_id"].astype(str).isin(existing_ids)]

    _append_rows_to_dataset(new_df, _load_dataset())

    full = _ensure_numeric_and_dates(_load_dataset())
    channel_df = full[full["channel_id"].astype(str) == str(channel_id)].copy()
    recent_df = channel_df[channel_df["video_publishedAt"] >= pd.Timestamp(cutoff)]
    title = _safe_get(channel, ["snippet", "title"], channel_id)
    return recent_df if not recent_df.empty else channel_df, "youtube_api", channel_id, str(title)


def _gemini_generate_text(gemini_key: str, model: str, prompt: str) -> str:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={gemini_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(endpoint, json=payload, timeout=90)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini text API error ({response.status_code}): {response.text[:500]}")
    body = response.json()
    texts: List[str] = []
    for candidate in body.get("candidates", []):
        for part in _safe_get(candidate, ["content", "parts"], []) or []:
            txt = part.get("text")
            if txt:
                texts.append(txt)
    if not texts:
        raise RuntimeError("Gemini did not return text output.")
    return "\n\n".join(texts)


def render() -> None:
    st.title("Ytuber")
    if build is None:
        st.error("Missing dependency: google-api-python-client. Install with: python3 -m pip install google-api-python-client")
        return
    st.write(
        "Live + cached YouTube channel intelligence. "
        "Checks local dataset first; fetches from YouTube API only when needed or forced."
    )

    youtube_api_key = st.text_input(
        "YouTube API Key",
        value=os.getenv("YOUTUBE_API_KEY", ""),
        type="password",
    )
    channel_query = st.text_input("Channel handle / name / channel ID", value="@veritasium")
    force_refresh = st.checkbox("Force API refresh (ignore cache)", value=False)

    analyze = st.button("Analyze Last 1 Year", type="primary", use_container_width=True)

    if analyze:
        if not youtube_api_key.strip():
            st.error("YouTube API key is required.")
            return
        if not channel_query.strip():
            st.error("Enter a channel handle or ID.")
            return

        with st.spinner("Loading channel data..."):
            try:
                channel_df, source, channel_id, channel_title = _fetch_or_get_cached_channel(
                    channel_query=channel_query.strip(),
                    youtube_api_key=youtube_api_key.strip(),
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                st.error(f"Failed to load channel data: {exc}")
                return

        st.session_state["ytuber_channel_df"] = channel_df
        st.session_state["ytuber_channel_title"] = channel_title
        st.session_state["ytuber_channel_id"] = channel_id
        st.session_state["ytuber_source"] = source

    if "ytuber_channel_df" not in st.session_state:
        st.info("Run analysis for a channel to see live stats and recommendations.")
        return

    channel_df = st.session_state["ytuber_channel_df"]
    channel_title = st.session_state.get("ytuber_channel_title", "")
    channel_id = st.session_state.get("ytuber_channel_id", "")
    source = st.session_state.get("ytuber_source", "")

    if channel_df.empty:
        st.warning("No videos available for this channel in the last year.")
        return

    st.success(f"Loaded `{channel_title}` ({channel_id}) from `{source}`")

    channel_df = _ensure_numeric_and_dates(channel_df)
    total_videos = len(channel_df)
    total_views = int(channel_df["views"].fillna(0).sum())
    total_likes = int(channel_df["likes"].fillna(0).sum())
    total_comments = int(channel_df["comments"].fillna(0).sum())
    avg_views = int(channel_df["views"].fillna(0).mean())
    med_eng = channel_df["engagement_rate"].median() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Videos (1Y)", f"{total_videos:,}")
    c2.metric("Total Views", f"{total_views:,}")
    c3.metric("Total Likes", f"{total_likes:,}")
    c4.metric("Total Comments", f"{total_comments:,}")
    c5.metric("Avg Views/Video", f"{avg_views:,}")
    c6.metric("Median Engagement", f"{med_eng:.2f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Video + Views Trend")
        trend = (
            channel_df.groupby("publish_month")
            .agg(videos=("video_id", "count"), views=("views", "sum"))
            .reset_index()
            .sort_values("publish_month")
        )
        st.line_chart(trend.set_index("publish_month")[["videos", "views"]], height=320)

    with right:
        st.subheader("Top 10 Videos (Last 1 Year)")
        top_videos = channel_df[
            ["video_title", "views", "likes", "comments", "engagement_rate", "video_publishedAt", "video_id"]
        ].sort_values("views", ascending=False).head(10)
        st.dataframe(top_videos, use_container_width=True)

    st.subheader("Detailed Analysis")
    best = channel_df.sort_values("views", ascending=False).head(20)
    top_keywords = []
    for t in best["video_title"].fillna("").astype(str).tolist():
        top_keywords.extend([w.lower() for w in t.replace("-", " ").split() if len(w) > 3])
    kw_series = pd.Series(top_keywords)
    keywords = kw_series.value_counts().head(10).index.tolist() if not kw_series.empty else []

    st.markdown(
        f"- Best-performing window: `{best['video_publishedAt'].min()}` to `{best['video_publishedAt'].max()}`\n"
        f"- Top repeated title keywords: `{', '.join(keywords) if keywords else 'N/A'}`\n"
        f"- Recommendation: target formats similar to top 20 videos and retain high-contrast thumbnail patterns."
    )

    st.markdown("---")
    st.subheader("Gemini Creative Studio")

    gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    text_model = st.text_input("Gemini text model", value="gemini-2.0-flash")
    image_model = st.text_input("Gemini image model", value="gemini-2.0-flash-preview-image-generation")

    creative_prompt = st.text_area(
        "Creative brief",
        value=(
            f"Channel: {channel_title}. Create high-performing content ideas for the next month based on this channel's recent stats."
        ),
        height=100,
    )

    col_a, col_b = st.columns(2)
    gen_text = col_a.button("Generate Titles/Descriptions/Scripts", use_container_width=True)
    gen_thumb = col_b.button("Generate Thumbnail Images", use_container_width=True)

    if gen_text:
        if not gemini_key.strip():
            st.error("Gemini API key is required for creative generation.")
        else:
            prompt = (
                f"You are a YouTube strategist. Based on this channel data summary:\n"
                f"Videos last year: {total_videos}, total views: {total_views}, avg views/video: {avg_views}, median engagement: {med_eng:.2f}%\n"
                f"Top title keywords: {', '.join(keywords)}\n"
                f"Brief: {creative_prompt}\n\n"
                "Return:\n"
                "1) 12 title ideas\n"
                "2) 5 full descriptions\n"
                "3) 3 short script outlines (hook, body bullets, CTA)\n"
                "4) 8 thumbnail concepts (visual direction, text overlay, color/subject)\n"
            )
            with st.spinner("Calling Gemini for creative assets..."):
                try:
                    creative_text = _gemini_generate_text(gemini_key.strip(), text_model.strip(), prompt)
                    st.text_area("Gemini Output", value=creative_text, height=420)
                except Exception as exc:
                    st.error(f"Gemini generation failed: {exc}")

    if gen_thumb:
        if not gemini_key.strip():
            st.error("Gemini API key is required for thumbnail generation.")
        else:
            base_title = channel_df.sort_values("views", ascending=False).head(1)["video_title"].iloc[0]
            with st.spinner("Generating thumbnails with Gemini image model..."):
                try:
                    generator = ThumbnailGenerator(provider="gemini", api_key=gemini_key.strip(), model=image_model.strip())
                    images = generator.generate(
                        title=f"Inspired by: {base_title}",
                        context=creative_prompt,
                        style="High contrast, clean focal subject, bold science visual, 16:9",
                        negative_prompt="low contrast, clutter, tiny text",
                        count=2,
                    )
                    out_dir = os.path.join("outputs", "thumbnails")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    for idx, generated in enumerate(images, start=1):
                        st.image(generated.image_bytes, caption=f"Ytuber Thumbnail {idx}")
                        ext = "png" if "png" in generated.mime_type else "jpg"
                        filename = f"ytuber_{channel_id}_{ts}_{idx}.{ext}"
                        with open(os.path.join(out_dir, filename), "wb") as fp:
                            fp.write(generated.image_bytes)
                        st.download_button(
                            label=f"Download thumbnail {idx}",
                            data=generated.image_bytes,
                            file_name=filename,
                            mime=generated.mime_type,
                            key=f"ytuber_dl_{idx}_{ts}",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"Thumbnail generation failed: {exc}")
