import base64
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageStat

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception

try:
    from pytrends.request import TrendReq
except Exception:
    TrendReq = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

from src.llm_integration.thumbnail_generator import ThumbnailGenerator


DATASET_PATH = os.path.join("data", "youtube api data", "research_science_channels_videos.csv")
BRAND_KIT_PATH = os.path.join("config", "brand_kit.json")
DEFAULT_CATEGORY = "research_science"
THUMB_KEYS = ["default", "medium", "high", "standard", "maxres"]
STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "for", "with", "on", "and", "or", "at", "is", "are", "was", "were",
    "this", "that", "how", "why", "what", "when", "from", "your", "you", "my", "we", "our", "it", "vs", "into",
}
POWER_WORDS = {
    "secret", "ultimate", "proven", "easy", "fast", "best", "shocking", "truth", "mistake", "science",
    "future", "breakthrough", "insane", "new", "critical", "warning", "guide", "explained", "hidden", "top",
}
POSITIVE_WORDS = {
    "great", "amazing", "awesome", "helpful", "love", "excellent", "perfect", "best", "clear", "fantastic", "cool",
}
NEGATIVE_WORDS = {
    "bad", "boring", "confusing", "worse", "worst", "hate", "useless", "clickbait", "slow", "poor", "terrible",
}


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
    max_videos: int = 600,
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


def _fetch_video_comments(youtube, video_id: str, max_comments: int = 100) -> List[str]:
    out: List[str] = []
    token: Optional[str] = None

    while len(out) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=min(100, max_comments - len(out)),
            pageToken=token,
            order="relevance",
        )
        resp = _api_call_with_backoff(req.execute)
        items = resp.get("items", [])
        for item in items:
            text = _safe_get(item, ["snippet", "topLevelComment", "snippet", "textDisplay"], "")
            if text:
                out.append(str(text))
        token = resp.get("nextPageToken")
        if not token:
            break
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


def _parse_iso_duration_seconds(duration: str) -> int:
    if not isinstance(duration, str):
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _ensure_numeric_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["views", "likes", "comments"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "video_publishedAt" in out.columns:
        out["video_publishedAt"] = pd.to_datetime(out["video_publishedAt"], errors="coerce", utc=True)
    out["engagement_rate"] = (out["likes"].fillna(0) + out["comments"].fillna(0)) / out["views"].clip(lower=1)
    out["publish_month"] = out["video_publishedAt"].dt.to_period("M").astype(str)
    out["publish_day"] = out["video_publishedAt"].dt.day_name()
    out["publish_hour"] = out["video_publishedAt"].dt.hour
    out["duration_seconds"] = out["duration"].fillna("").astype(str).map(_parse_iso_duration_seconds)
    out["is_short"] = out["duration_seconds"] <= 60
    out["title_length"] = out["video_title"].fillna("").astype(str).str.len()
    return out


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def _top_keywords(df: pd.DataFrame, top_n: int = 30) -> List[str]:
    counter: Counter = Counter()
    for title in df["video_title"].fillna("").astype(str):
        counter.update(_tokenize(title))
    return [k for k, _ in counter.most_common(top_n)]


def _keyword_intel(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for _, row in df.iterrows():
        title = str(row.get("video_title", ""))
        views = float(row.get("views") or 0)
        eng = float(row.get("engagement_rate") or 0)
        published = row.get("video_publishedAt")
        recency_weight = 1.0
        if pd.notna(published):
            days = (now - published.to_pydatetime()).days
            recency_weight = max(0.1, 1 - min(days / 365, 0.9))

        seen = set(_tokenize(title))
        for token in seen:
            rows.append(
                {
                    "keyword": token,
                    "views": views,
                    "engagement": eng,
                    "recency_weight": recency_weight,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["keyword", "videos", "avg_views", "avg_engagement", "momentum", "score"])

    kdf = pd.DataFrame(rows)
    out = (
        kdf.groupby("keyword", dropna=False)
        .agg(
            videos=("keyword", "count"),
            avg_views=("views", "mean"),
            avg_engagement=("engagement", "mean"),
            momentum=("recency_weight", "mean"),
        )
        .reset_index()
    )

    if out.empty:
        return out

    max_views = max(out["avg_views"].max(), 1)
    max_eng = max(out["avg_engagement"].max(), 0.0001)
    max_momentum = max(out["momentum"].max(), 0.0001)
    competition_proxy = out["videos"] / max(out["videos"].max(), 1)

    out["score"] = (
        (out["avg_views"] / max_views) * 40
        + (out["avg_engagement"] / max_eng) * 30
        + (out["momentum"] / max_momentum) * 20
        + (1 - competition_proxy) * 10
    )

    return out.sort_values("score", ascending=False).head(top_n)


def _title_score(title: str, keyword_hints: List[str]) -> Tuple[int, Dict[str, int], List[str]]:
    text = title.strip()
    lower = text.lower()

    length = len(text)
    word_count = len(text.split())

    length_score = max(0, 30 - int(abs(length - 55) * 0.8))
    clarity_score = 20 if 6 <= word_count <= 12 else max(6, 20 - abs(word_count - 9) * 2)
    number_score = 12 if re.search(r"\d", text) else 0
    curiosity_score = 10 if any(p in text for p in ["?", "!", ":"]) else 0
    power_score = min(15, sum(1 for w in POWER_WORDS if w in lower) * 5)
    keyword_matches = sum(1 for k in keyword_hints[:12] if k and k in lower)
    keyword_score = min(13, keyword_matches * 3)

    total = max(0, min(100, length_score + clarity_score + number_score + curiosity_score + power_score + keyword_score))

    suggestions: List[str] = []
    if length < 40:
        suggestions.append("Make title more specific; 45-65 chars usually performs better.")
    if length > 70:
        suggestions.append("Shorten title for stronger mobile readability.")
    if number_score == 0:
        suggestions.append("Consider adding a number or quantified claim.")
    if curiosity_score == 0:
        suggestions.append("Try adding a curiosity trigger such as '?' or a strong contrast.")
    if power_score == 0:
        suggestions.append("Use at least one strong power word (e.g., proven, hidden, breakthrough).")
    if keyword_score == 0 and keyword_hints:
        suggestions.append(f"Include one of your high-opportunity keywords: {', '.join(keyword_hints[:5])}.")

    parts = {
        "Length": int(length_score),
        "Clarity": int(clarity_score),
        "Numbers": int(number_score),
        "Curiosity": int(curiosity_score),
        "Power Words": int(power_score),
        "Keyword Match": int(keyword_score),
    }
    return total, parts, suggestions


def _description_score(description: str, keyword_hints: List[str]) -> Tuple[int, Dict[str, int], List[str]]:
    text = description.strip()
    lower = text.lower()
    length = len(text)

    length_score = 30 if 400 <= length <= 1800 else max(6, 30 - int(abs(length - 900) / 80))
    cta_score = 20 if any(x in lower for x in ["subscribe", "comment", "watch", "join", "follow", "link"]) else 5
    keyword_matches = sum(1 for k in keyword_hints[:15] if k in lower)
    keyword_score = min(25, keyword_matches * 4)
    structure_score = 15 if "\n" in text else 8
    hashtags = re.findall(r"#\w+", text)
    hashtag_score = 10 if 1 <= len(hashtags) <= 3 else 4

    total = max(0, min(100, length_score + cta_score + keyword_score + structure_score + hashtag_score))

    tips = []
    if length < 300:
        tips.append("Description is short; add context, value bullets, and timestamps if possible.")
    if cta_score < 20:
        tips.append("Add a clear CTA (subscribe, watch next, comment).")
    if keyword_score < 8 and keyword_hints:
        tips.append(f"Add relevant keywords: {', '.join(keyword_hints[:5])}.")
    if hashtag_score < 10:
        tips.append("Use 1-3 relevant hashtags, not more.")

    parts = {
        "Length": int(length_score),
        "CTA": int(cta_score),
        "Keywords": int(keyword_score),
        "Structure": int(structure_score),
        "Hashtags": int(hashtag_score),
    }
    return total, parts, tips


def _compute_channel_audit(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ordered = df.sort_values("video_publishedAt").copy()

    out["videos"] = len(ordered)
    out["median_views"] = float(ordered["views"].median()) if not ordered.empty else 0
    out["avg_engagement"] = float(ordered["engagement_rate"].mean()) if not ordered.empty else 0
    out["shorts_ratio"] = float(ordered["is_short"].mean()) if not ordered.empty else 0

    if len(ordered) > 2:
        gaps = ordered["video_publishedAt"].diff().dt.total_seconds().dropna() / 86400
        mean_gap = float(gaps.mean()) if not gaps.empty else 0
        std_gap = float(gaps.std()) if not gaps.empty else 0
        consistency = max(0.0, 100 - (std_gap / max(mean_gap, 1)) * 40)
    else:
        mean_gap = 0
        consistency = 0

    out["avg_upload_gap_days"] = mean_gap
    out["consistency_score"] = float(consistency)

    recent_cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    previous_cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    recent = ordered[ordered["video_publishedAt"] >= recent_cutoff]
    previous = ordered[(ordered["video_publishedAt"] < recent_cutoff) & (ordered["video_publishedAt"] >= previous_cutoff)]

    recent_avg = float(recent["views"].mean()) if not recent.empty else 0
    previous_avg = float(previous["views"].mean()) if not previous.empty else 0
    growth = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
    out["view_growth_90d_pct"] = growth

    threshold = max(float(ordered["views"].median()) * 2.0, 1.0)
    out["outlier_rate"] = float((ordered["views"] >= threshold).mean()) if not ordered.empty else 0

    return out


def _load_brand_kit() -> Dict[str, Any]:
    default = {
        "brand_name": "Ayush Creator Lab",
        "tone": "Direct, curious, insight-driven",
        "audience": "Ambitious learners and creators",
        "visual_style": "High contrast, bold focal subject, clean scientific style",
        "banned_words": ["cheap", "fake", "impossible"],
        "cta_style": "Invite viewers to test and comment",
    }
    if not os.path.exists(BRAND_KIT_PATH):
        return default
    try:
        with open(BRAND_KIT_PATH, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        merged = default.copy()
        merged.update(data)
        return merged
    except Exception:
        return default


def _save_brand_kit(kit: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(BRAND_KIT_PATH), exist_ok=True)
    with open(BRAND_KIT_PATH, "w", encoding="utf-8") as fp:
        json.dump(kit, fp, indent=2)


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

    video_ids = _fetch_recent_video_ids(youtube, uploads_pid, cutoff, max_videos=600)
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


def _gemini_vision_review(gemini_key: str, model: str, image_bytes: bytes, prompt: str) -> str:
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={gemini_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ]
    }
    response = requests.post(endpoint, json=payload, timeout=90)
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini vision API error ({response.status_code}): {response.text[:500]}")
    body = response.json()

    texts: List[str] = []
    for candidate in body.get("candidates", []):
        for part in _safe_get(candidate, ["content", "parts"], []) or []:
            txt = part.get("text")
            if txt:
                texts.append(txt)
    if not texts:
        raise RuntimeError("Gemini vision returned no text output.")
    return "\n\n".join(texts)


def _render_overview(channel_df: pd.DataFrame) -> None:
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
        st.subheader("Top 12 Videos")
        top_videos = channel_df[
            ["video_title", "views", "likes", "comments", "engagement_rate", "video_publishedAt", "video_id"]
        ].sort_values("views", ascending=False).head(12)
        st.dataframe(top_videos, use_container_width=True)


def _render_channel_audit(channel_df: pd.DataFrame) -> None:
    st.subheader("Channel Audit")
    audit = _compute_channel_audit(channel_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consistency Score", f"{audit['consistency_score']:.1f}/100")
    c2.metric("Avg Upload Gap", f"{audit['avg_upload_gap_days']:.1f} days")
    c3.metric("90d View Growth", f"{audit['view_growth_90d_pct']:.1f}%")
    c4.metric("Outlier Rate", f"{audit['outlier_rate'] * 100:.1f}%")

    st.markdown("**Audit Notes**")
    notes = []
    if audit["consistency_score"] < 45:
        notes.append("Upload cadence is inconsistent. Use a fixed weekly posting pattern.")
    if audit["view_growth_90d_pct"] < 0:
        notes.append("Views are down vs previous 90 days. Test stronger hooks and tighter titles.")
    if audit["outlier_rate"] < 0.08:
        notes.append("Few breakout videos detected. Increase experimentation with bold concepts.")
    if audit["shorts_ratio"] > 0.7:
        notes.append("Channel is heavily shorts-weighted; blend long-form to deepen session time.")
    if not notes:
        notes.append("Performance is stable. Focus on scaling repeatable winning formats.")

    for item in notes:
        st.write(f"- {item}")


def _render_keyword_intel(channel_df: pd.DataFrame) -> List[str]:
    st.subheader("Keyword Intelligence")
    intel = _keyword_intel(channel_df)
    if intel.empty:
        st.info("Not enough text data to compute keyword insights.")
        return []

    st.dataframe(intel, use_container_width=True)
    top10 = intel.head(10)["keyword"].tolist()
    st.markdown("**High-opportunity keywords:** " + ", ".join(top10))
    return intel["keyword"].tolist()


def _render_title_seo_lab(keyword_hints: List[str]) -> None:
    st.subheader("Title & SEO Lab")
    test_title = st.text_input("Test title", value="The Hidden Physics Trick That Changes Everything")
    test_desc = st.text_area(
        "Test description",
        value="In this video we break down the science, show real examples, and explain how to apply this idea. Subscribe for more! #science #learning",
        height=120,
    )

    title_score, parts, tips = _title_score(test_title, keyword_hints)
    st.metric("Title Score", f"{title_score}/100")
    st.progress(min(max(title_score, 0), 100) / 100)
    st.dataframe(pd.DataFrame([parts]), use_container_width=True)
    for t in tips:
        st.write(f"- {t}")

    st.markdown("---")

    desc_score, desc_parts, desc_tips = _description_score(test_desc, keyword_hints)
    st.metric("Description Score", f"{desc_score}/100")
    st.progress(min(max(desc_score, 0), 100) / 100)
    st.dataframe(pd.DataFrame([desc_parts]), use_container_width=True)
    for t in desc_tips:
        st.write(f"- {t}")


def _render_title_ab_lab(keyword_hints: List[str], gemini_key: str, text_model: str) -> None:
    st.subheader("Title A/B Lab")
    base_title = st.text_input("Base title", value="This Physics Experiment Broke My Brain")
    variant_count = st.slider("Variants", min_value=5, max_value=30, value=12)

    run = st.button("Generate A/B Variants", use_container_width=True)
    if not run:
        return

    variants: List[str] = []
    if gemini_key.strip():
        prompt = (
            f"Generate {variant_count} distinct YouTube title variants for this base title:\n"
            f"{base_title}\n\n"
            f"Use these keywords when relevant: {', '.join(keyword_hints[:15])}\n"
            "Return one title per line only."
        )
        try:
            raw = _gemini_generate_text(gemini_key.strip(), text_model.strip(), prompt)
            lines = [ln.strip(" -0123456789.") for ln in raw.splitlines() if ln.strip()]
            variants = list(dict.fromkeys(lines))
        except Exception as exc:
            st.warning(f"Gemini title generation failed; using heuristic variants. {exc}")

    if not variants:
        suffixes = [
            "(Explained)", "in 10 Minutes", "No One Talks About This", "What Happens Next?", "The Science Behind It",
            "Without The Hype", "The Brutal Truth", "Step by Step", "Beginner to Pro", "Before It’s Too Late",
        ]
        variants = [f"{base_title} {suffixes[i % len(suffixes)]}" for i in range(variant_count)]

    rows = []
    for t in variants[:variant_count]:
        score, parts, _tips = _title_score(t, keyword_hints)
        rows.append({"title": t, "score": score, **parts})

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    st.dataframe(out, use_container_width=True)


def _render_competitor_benchmark(youtube_api_key: str) -> None:
    st.subheader("Competitor Benchmark")
    handles = st.text_area(
        "Competitor handles (comma separated)",
        value="@3blue1brown,@veritasium,@smartereveryday",
        height=90,
    )

    run = st.button("Run Competitor Benchmark", use_container_width=True)
    if not run:
        st.caption("Enter competitor handles and run benchmark.")
        return

    if not youtube_api_key.strip():
        st.error("YouTube API key required for competitor benchmarking.")
        return

    competitors = [h.strip() for h in handles.split(",") if h.strip()]
    rows = []

    with st.spinner("Loading competitor channels..."):
        for handle in competitors:
            try:
                cdf, source, cid, title = _fetch_or_get_cached_channel(handle, youtube_api_key.strip(), force_refresh=False)
                cdf = _ensure_numeric_and_dates(cdf)
                rows.append(
                    {
                        "handle": handle,
                        "channel_title": title,
                        "channel_id": cid,
                        "source": source,
                        "videos_1y": len(cdf),
                        "total_views": int(cdf["views"].fillna(0).sum()) if not cdf.empty else 0,
                        "avg_views": int(cdf["views"].fillna(0).mean()) if not cdf.empty else 0,
                        "median_engagement": float(cdf["engagement_rate"].median()) if not cdf.empty else 0.0,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "handle": handle,
                        "channel_title": "ERROR",
                        "channel_id": "",
                        "source": "error",
                        "videos_1y": 0,
                        "total_views": 0,
                        "avg_views": 0,
                        "median_engagement": 0.0,
                        "error": str(exc),
                    }
                )

    if not rows:
        st.warning("No competitor data produced.")
        return

    bdf = pd.DataFrame(rows).sort_values("total_views", ascending=False)
    st.dataframe(bdf, use_container_width=True)


def _render_content_gap_finder(channel_df: pd.DataFrame, youtube_api_key: str) -> None:
    st.subheader("Content Gap Finder")
    handles = st.text_area(
        "Compare against competitors (comma separated)",
        value="@3blue1brown,@smartereveryday,@RealEngineering",
        height=80,
    )
    run = st.button("Find Content Gaps", use_container_width=True)
    if not run:
        return

    if not youtube_api_key.strip():
        st.error("YouTube API key required.")
        return

    base_keywords = set(_top_keywords(channel_df, 80))
    competitors = [h.strip() for h in handles.split(",") if h.strip()]

    rows = []
    with st.spinner("Analyzing competitor topic coverage..."):
        for handle in competitors:
            try:
                cdf, _source, _cid, ctitle = _fetch_or_get_cached_channel(handle, youtube_api_key.strip(), force_refresh=False)
                cdf = _ensure_numeric_and_dates(cdf)
                intel = _keyword_intel(cdf, top_n=60)
                for _, r in intel.iterrows():
                    kw = str(r["keyword"])
                    if kw in base_keywords:
                        continue
                    rows.append(
                        {
                            "keyword": kw,
                            "competitor": ctitle,
                            "opportunity_score": float(r["score"]),
                            "avg_views": float(r["avg_views"]),
                            "avg_engagement": float(r["avg_engagement"]),
                        }
                    )
            except Exception:
                continue

    if not rows:
        st.info("No clear gaps found from selected competitors.")
        return

    gdf = pd.DataFrame(rows)
    out = (
        gdf.groupby("keyword", dropna=False)
        .agg(
            opportunity_score=("opportunity_score", "mean"),
            avg_views=("avg_views", "mean"),
            avg_engagement=("avg_engagement", "mean"),
            competitor_count=("competitor", "nunique"),
        )
        .reset_index()
        .sort_values("opportunity_score", ascending=False)
        .head(40)
    )
    st.dataframe(out, use_container_width=True)


def _render_trend_radar(channel_df: pd.DataFrame) -> None:
    st.subheader("Trend Radar")
    now = datetime.now(timezone.utc)
    recent_60 = channel_df[channel_df["video_publishedAt"] >= (now - timedelta(days=60))]
    prev_60 = channel_df[
        (channel_df["video_publishedAt"] < (now - timedelta(days=60)))
        & (channel_df["video_publishedAt"] >= (now - timedelta(days=120)))
    ]

    def keyword_counter(frame: pd.DataFrame) -> Counter:
        c = Counter()
        for title in frame["video_title"].fillna("").astype(str):
            c.update(set(_tokenize(title)))
        return c

    c_recent = keyword_counter(recent_60)
    c_prev = keyword_counter(prev_60)

    rows = []
    for kw, recent_count in c_recent.items():
        prev_count = c_prev.get(kw, 0)
        growth = recent_count - prev_count
        rows.append(
            {
                "keyword": kw,
                "recent_mentions": recent_count,
                "previous_mentions": prev_count,
                "momentum_delta": growth,
            }
        )

    tdf = pd.DataFrame(rows)
    if tdf.empty:
        st.info("Not enough recent data for trend radar.")
        return

    tdf = tdf.sort_values(["momentum_delta", "recent_mentions"], ascending=[False, False]).head(25)
    st.dataframe(tdf, use_container_width=True)


def _render_trend_api(keyword_hints: List[str]) -> None:
    st.subheader("Trend APIs")
    seed = ", ".join(keyword_hints[:5]) if keyword_hints else "science, physics, engineering"
    keywords = st.text_input("Trend keywords (comma separated, max 5)", value=seed)
    news_api_key = st.text_input("News API key (optional)", value=os.getenv("NEWSAPI_KEY", ""), type="password")

    klist = [k.strip() for k in keywords.split(",") if k.strip()][:5]
    c1, c2 = st.columns(2)
    run_trends = c1.button("Fetch Google Trends", use_container_width=True)
    run_news = c2.button("Fetch News Signals", use_container_width=True)

    if run_trends:
        if TrendReq is None:
            st.error("pytrends unavailable in this environment.")
        elif not klist:
            st.error("Add at least one keyword.")
        else:
            try:
                pytrend = TrendReq(hl="en-US", tz=360)
                pytrend.build_payload(klist, timeframe="today 12-m")
                iot = pytrend.interest_over_time().drop(columns=["isPartial"], errors="ignore")
                st.line_chart(iot)
                rq = pytrend.related_queries()
                primary = klist[0]
                top_q = _safe_get(rq, [primary, "top"], None)
                if top_q is not None and not top_q.empty:
                    st.markdown("**Related Queries**")
                    st.dataframe(top_q.head(20), use_container_width=True)
            except Exception as exc:
                st.error(f"Google Trends fetch failed: {exc}")

    if run_news:
        if not news_api_key.strip():
            st.error("News API key is required.")
        elif not klist:
            st.error("Add at least one keyword.")
        else:
            query = " OR ".join(klist)
            url = (
                "https://newsapi.org/v2/everything?"
                f"q={requests.utils.quote(query)}&sortBy=publishedAt&language=en&pageSize=20&apiKey={news_api_key.strip()}"
            )
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code >= 400:
                    raise RuntimeError(resp.text[:500])
                body = resp.json()
                articles = body.get("articles", [])
                ndf = pd.DataFrame(
                    [
                        {
                            "title": a.get("title", ""),
                            "source": _safe_get(a, ["source", "name"], ""),
                            "publishedAt": a.get("publishedAt", ""),
                            "url": a.get("url", ""),
                        }
                        for a in articles
                    ]
                )
                if ndf.empty:
                    st.info("No recent news found for these keywords.")
                else:
                    st.dataframe(ndf, use_container_width=True)
            except Exception as exc:
                st.error(f"News API fetch failed: {exc}")


def _render_content_planner(channel_df: pd.DataFrame) -> None:
    st.subheader("Content Planner")

    day_perf = (
        channel_df.groupby("publish_day", dropna=False)
        .agg(avg_views=("views", "mean"), median_engagement=("engagement_rate", "median"), videos=("video_id", "count"))
        .reset_index()
        .sort_values("avg_views", ascending=False)
    )

    hour_perf = (
        channel_df.groupby("publish_hour", dropna=False)
        .agg(avg_views=("views", "mean"), median_engagement=("engagement_rate", "median"), videos=("video_id", "count"))
        .reset_index()
        .sort_values("avg_views", ascending=False)
    )

    best_day = day_perf.iloc[0]["publish_day"] if not day_perf.empty else "Wednesday"
    best_hour = int(hour_perf.iloc[0]["publish_hour"]) if not hour_perf.empty else 15

    c1, c2 = st.columns(2)
    c1.metric("Best Publishing Day", str(best_day))
    c2.metric("Best Publishing Hour (UTC)", f"{best_hour:02d}:00")

    st.markdown("**Day Performance**")
    st.dataframe(day_perf, use_container_width=True)

    st.markdown("**Hour Performance (UTC)**")
    st.dataframe(hour_perf.head(12), use_container_width=True)

    top_topics = _top_keywords(channel_df, top_n=12)
    st.markdown("**Suggested next content angles:** " + ", ".join(top_topics[:8]))

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_map = {d: i for i, d in enumerate(weekday_order)}
    target_weekday = weekday_map.get(best_day, 2)

    start = datetime.now(timezone.utc)
    plan_rows = []
    cursor = start
    for i in range(1, 5):
        while cursor.weekday() != target_weekday:
            cursor += timedelta(days=1)
        plan_rows.append(
            {
                "week": f"Week {i}",
                "publish_date_utc": cursor.date().isoformat(),
                "publish_time_utc": f"{best_hour:02d}:00",
                "topic_hint": top_topics[(i - 1) % max(len(top_topics), 1)] if top_topics else "core topic",
            }
        )
        cursor += timedelta(days=7)

    st.markdown("**4-Week Suggested Calendar**")
    st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)


def _render_thumbnail_critic(gemini_key: str, image_model: str) -> None:
    st.subheader("Thumbnail Critic")
    uploaded = st.file_uploader("Upload thumbnail image", type=["png", "jpg", "jpeg"], key="thumb_critic_upload")
    if not uploaded:
        st.caption("Upload an image to run quality diagnostics.")
        return

    image_bytes = uploaded.read()
    st.image(image_bytes, caption="Uploaded Thumbnail", use_container_width=True)

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        stat = ImageStat.Stat(img)
        brightness = sum(stat.mean) / 3
        contrast = sum(stat.stddev) / 3
        w, h = img.size
        aspect = w / max(h, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Brightness", f"{brightness:.1f}")
        c2.metric("Contrast", f"{contrast:.1f}")
        c3.metric("Aspect Ratio", f"{aspect:.2f}")

        quick_tips = []
        if brightness < 75:
            quick_tips.append("Image is dark; increase exposure/highlights for mobile readability.")
        if contrast < 45:
            quick_tips.append("Contrast is low; increase foreground/background separation.")
        if aspect < 1.6 or aspect > 1.9:
            quick_tips.append("Use 16:9 framing for YouTube thumbnails.")
        if not quick_tips:
            quick_tips.append("Technical baseline looks good. Focus on emotional hook and subject clarity.")

        for t in quick_tips:
            st.write(f"- {t}")
    except Exception as exc:
        st.warning(f"Could not compute local image metrics: {exc}")

    run_ai = st.button("Run Gemini Visual Critique", use_container_width=True)
    if run_ai:
        if not gemini_key.strip():
            st.error("Gemini API key required for visual critique.")
            return
        prompt = (
            "Review this YouTube thumbnail. Provide: 1) CTR potential score out of 10, "
            "2) readability feedback, 3) emotional impact feedback, 4) composition fixes, "
            "5) improved prompt to regenerate a better version."
        )
        try:
            review = _gemini_vision_review(gemini_key.strip(), image_model.strip(), image_bytes, prompt)
            st.text_area("Gemini Visual Critique", value=review, height=320)
        except Exception as exc:
            st.error(f"Gemini visual critique failed: {exc}")


def _render_comment_intelligence(channel_df: pd.DataFrame, youtube_api_key: str) -> None:
    st.subheader("Comment Intelligence")
    top_n_videos = st.slider("Top videos to inspect", 1, 10, 5)
    max_comments = st.slider("Max comments per video", 20, 200, 80)

    run = st.button("Analyze Comments", use_container_width=True)
    if not run:
        return

    if not youtube_api_key.strip():
        st.error("YouTube API key required for comment intelligence.")
        return

    top_videos = channel_df.sort_values("views", ascending=False).head(top_n_videos)
    youtube = _yt_client(youtube_api_key.strip())

    comments: List[str] = []
    with st.spinner("Fetching comments from YouTube API..."):
        for _, row in top_videos.iterrows():
            vid = str(row.get("video_id", ""))
            if not vid:
                continue
            try:
                comments.extend(_fetch_video_comments(youtube, vid, max_comments=max_comments))
            except Exception:
                continue

    if not comments:
        st.info("No comments pulled (comments may be disabled for selected videos).")
        return

    token_counter = Counter()
    pos = neg = questions = 0
    for c in comments:
        toks = _tokenize(c)
        token_counter.update(toks)
        if any(w in c.lower() for w in POSITIVE_WORDS):
            pos += 1
        if any(w in c.lower() for w in NEGATIVE_WORDS):
            neg += 1
        if "?" in c:
            questions += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments Analyzed", f"{len(comments):,}")
    c2.metric("Positive Signal", f"{(pos / len(comments) * 100):.1f}%")
    c3.metric("Negative Signal", f"{(neg / len(comments) * 100):.1f}%")
    c4.metric("Questions", f"{questions:,}")

    top_terms = pd.DataFrame(token_counter.most_common(30), columns=["term", "count"])
    st.markdown("**Most frequent comment terms**")
    st.dataframe(top_terms, use_container_width=True)


def _render_transcript_lab(channel_df: pd.DataFrame, gemini_key: str, text_model: str) -> None:
    st.subheader("Transcript Lab")

    if YouTubeTranscriptApi is None:
        st.warning("`youtube-transcript-api` is not installed in this environment.")
        return

    choices = channel_df.sort_values("views", ascending=False).head(30)
    options = {
        f"{row['video_title'][:80]} ({row['video_id']})": str(row["video_id"])
        for _, row in choices.iterrows()
    }
    if not options:
        st.info("No videos available for transcript analysis.")
        return

    selected_label = st.selectbox("Select video for transcript", list(options.keys()))
    video_id = options[selected_label]

    run = st.button("Fetch Transcript", use_container_width=True)
    if not run:
        return

    try:
        transcript_items = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([it.get("text", "") for it in transcript_items])
    except Exception as exc:
        st.error(f"Transcript fetch failed: {exc}")
        return

    st.metric("Transcript Words", f"{len(transcript_text.split()):,}")
    st.text_area("Transcript Preview", transcript_text[:8000], height=220)

    if gemini_key.strip():
        analyze = st.button("Generate Retention & Script Improvements", use_container_width=True)
        if analyze:
            prompt = (
                "Analyze this YouTube transcript and return:\n"
                "1) Weak retention zones\n"
                "2) Suggested hook rewrite\n"
                "3) 5 concise pacing improvements\n"
                "4) Strong CTA rewrite\n\n"
                f"Transcript:\n{transcript_text[:25000]}"
            )
            try:
                analysis = _gemini_generate_text(gemini_key.strip(), text_model.strip(), prompt)
                st.text_area("Transcript Analysis", value=analysis, height=360)
            except Exception as exc:
                st.error(f"Gemini transcript analysis failed: {exc}")


def _render_brand_kit_manager() -> Dict[str, Any]:
    st.subheader("Brand Kit")
    kit = _load_brand_kit()

    brand_name = st.text_input("Brand name", value=str(kit.get("brand_name", "")))
    tone = st.text_input("Tone", value=str(kit.get("tone", "")))
    audience = st.text_input("Audience", value=str(kit.get("audience", "")))
    visual_style = st.text_area("Visual style", value=str(kit.get("visual_style", "")), height=90)
    banned_words = st.text_input("Banned words (comma separated)", value=", ".join(kit.get("banned_words", [])))
    cta_style = st.text_input("CTA style", value=str(kit.get("cta_style", "")))

    if st.button("Save Brand Kit", use_container_width=True):
        updated = {
            "brand_name": brand_name,
            "tone": tone,
            "audience": audience,
            "visual_style": visual_style,
            "banned_words": [w.strip() for w in banned_words.split(",") if w.strip()],
            "cta_style": cta_style,
        }
        _save_brand_kit(updated)
        st.success("Brand kit saved.")
        kit = updated

    return kit


def _render_ai_studio(
    channel_df: pd.DataFrame,
    channel_title: str,
    channel_id: str,
    keyword_hints: List[str],
    brand_kit: Dict[str, Any],
) -> None:
    st.subheader("AI Studio")

    gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    text_model = st.text_input("Gemini text model", value="gemini-2.0-flash")
    image_model = st.text_input("Gemini image model", value="gemini-2.0-flash-preview-image-generation")

    creative_brief = st.text_area(
        "Creative brief",
        value=f"Channel: {channel_title}. Create high-performing science content for next month.",
        height=100,
    )

    output_type = st.selectbox(
        "Creative task",
        [
            "Full Pack (titles + descriptions + scripts + thumbnail concepts)",
            "Titles Only",
            "Descriptions Only",
            "Scripts Only",
            "Hooks + CTAs",
        ],
        index=0,
    )

    col_a, col_b = st.columns(2)
    gen_text = col_a.button("Generate AI Content", use_container_width=True)
    gen_thumb = col_b.button("Generate Thumbnail Images", use_container_width=True)

    total_videos = len(channel_df)
    total_views = int(channel_df["views"].fillna(0).sum())
    avg_views = int(channel_df["views"].fillna(0).mean())
    med_eng = float(channel_df["engagement_rate"].median() * 100)

    if gen_text:
        if not gemini_key.strip():
            st.error("Gemini API key is required for AI content.")
        else:
            prompt = (
                "You are an advanced YouTube strategist. Produce concise, high-performing outputs.\n"
                f"Channel: {channel_title} ({channel_id})\n"
                f"Videos(1y): {total_videos}, Total views: {total_views}, Avg views/video: {avg_views}, Median engagement: {med_eng:.2f}%\n"
                f"Priority keywords: {', '.join(keyword_hints[:15])}\n"
                f"Brand Kit: {json.dumps(brand_kit)}\n"
                f"Task: {output_type}\n"
                f"Brief: {creative_brief}\n"
            )
            try:
                output = _gemini_generate_text(gemini_key.strip(), text_model.strip(), prompt)
                st.text_area("AI Output", value=output, height=460)
            except Exception as exc:
                st.error(f"Gemini generation failed: {exc}")

    if gen_thumb:
        if not gemini_key.strip():
            st.error("Gemini API key is required for thumbnail generation.")
        else:
            base_title = channel_df.sort_values("views", ascending=False).head(1)["video_title"].iloc[0]
            try:
                generator = ThumbnailGenerator(provider="gemini", api_key=gemini_key.strip(), model=image_model.strip())
                images = generator.generate(
                    title=f"Inspired by: {base_title}",
                    context=creative_brief,
                    style=str(brand_kit.get("visual_style", "High contrast, one clear subject, 16:9 composition")),
                    negative_prompt="clutter, tiny text, low contrast",
                    count=3,
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
                        key=f"ytuber_thumb_{idx}_{ts}",
                        use_container_width=True,
                    )
            except Exception as exc:
                st.error(f"Thumbnail generation failed: {exc}")


def render() -> None:
    st.title("Ytuber")
    if build is None:
        st.error("Missing dependency: google-api-python-client. Install with: python3 -m pip install google-api-python-client")
        return

    st.caption(
        "Creator Suite: cache-aware channel sync, analytics, SEO tooling, competitor tracking, trend APIs, transcript lab, comments intelligence, and AI studio."
    )

    youtube_api_key = st.text_input("YouTube API Key", value=os.getenv("YOUTUBE_API_KEY", ""), type="password")
    channel_query = st.text_input("Channel handle / name / channel ID", value="@veritasium")
    force_refresh = st.checkbox("Force API refresh (ignore cache)", value=False)

    analyze = st.button("Load Channel (Last 1 Year)", type="primary", use_container_width=True)

    if analyze:
        if not youtube_api_key.strip():
            st.error("YouTube API key is required.")
            return
        if not channel_query.strip():
            st.error("Enter a channel handle or ID.")
            return

        with st.spinner("Loading channel data from cache/API..."):
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
        st.info("Load a channel to unlock the full Ytuber suite.")
        return

    channel_df = st.session_state["ytuber_channel_df"]
    channel_title = st.session_state.get("ytuber_channel_title", "")
    channel_id = st.session_state.get("ytuber_channel_id", "")
    source = st.session_state.get("ytuber_source", "")

    if channel_df.empty:
        st.warning("No videos available for this channel in the last year.")
        return

    channel_df = _ensure_numeric_and_dates(channel_df)
    st.success(f"Loaded `{channel_title}` ({channel_id}) from `{source}`")

    tabs = st.tabs(
        [
            "Overview",
            "Channel Audit",
            "Keyword Intel",
            "Title & SEO Lab",
            "Title A/B Lab",
            "Competitor Benchmark",
            "Content Gap Finder",
            "Trend Radar",
            "Trend APIs",
            "Comment Intelligence",
            "Transcript Lab",
            "Content Planner",
            "Thumbnail Critic",
            "Brand Kit",
            "AI Studio",
        ]
    )

    with tabs[0]:
        _render_overview(channel_df)

    with tabs[1]:
        _render_channel_audit(channel_df)

    with tabs[2]:
        keyword_hints = _render_keyword_intel(channel_df)
        st.session_state["ytuber_keyword_hints"] = keyword_hints

    with tabs[3]:
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        _render_title_seo_lab(hints)

    with tabs[4]:
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        _render_title_ab_lab(hints, os.getenv("GEMINI_API_KEY", ""), "gemini-2.0-flash")

    with tabs[5]:
        _render_competitor_benchmark(youtube_api_key)

    with tabs[6]:
        _render_content_gap_finder(channel_df, youtube_api_key)

    with tabs[7]:
        _render_trend_radar(channel_df)

    with tabs[8]:
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        _render_trend_api(hints)

    with tabs[9]:
        _render_comment_intelligence(channel_df, youtube_api_key)

    with tabs[10]:
        _render_transcript_lab(channel_df, os.getenv("GEMINI_API_KEY", ""), "gemini-2.0-flash")

    with tabs[11]:
        _render_content_planner(channel_df)

    with tabs[12]:
        _render_thumbnail_critic(os.getenv("GEMINI_API_KEY", ""), "gemini-2.0-flash")

    with tabs[13]:
        brand_kit = _render_brand_kit_manager()
        st.session_state["ytuber_brand_kit"] = brand_kit

    with tabs[14]:
        hints = st.session_state.get("ytuber_keyword_hints") or _top_keywords(channel_df, 20)
        brand_kit = st.session_state.get("ytuber_brand_kit") or _load_brand_kit()
        _render_ai_studio(channel_df, channel_title, channel_id, hints, brand_kit)
