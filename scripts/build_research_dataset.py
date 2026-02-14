import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ============================================================
# TEAM CONFIG (only edit these two blocks)
# ============================================================

CATEGORY_NAME = "research_science"

CHANNEL_HANDLES = [
    "@veritasium",
    "@3blue1brown",
    "@AsapSCIENCE",
    "@SciShow",
    "@PBSSpaceTime",
    "@Seeker",
    "@NileRed",
    "@NileBlue",
    "@smartereveryday",
    "@RealEngineering",
    "@PracticalEngineeringChannel",
    "@numberphile",
    "@Computerphile",
    "@minutephysics",
    "@standupmaths",
    "@SteveMould",
    "@physicsgirl",
    "@SabineHossenfelder",
    "@DomainofScience",
    "@journeytomicro",
    "@kurzgesagt",
    "@ben_eater",
    "@ArvinAsh",
    "@statquest",
    "@TwoMinutePapers",
    "@AstrumSpace",
    "@fermilab",
    "@SixtySymbols",
    "@periodicvideos",
    "@DrBenMiles",
    "@medlifecrisis",
    "@sciencephiletheai",
    "@AlphaPhoenixChannel",
    "@CoolWorldsLab",
    "@CleoAbram",
    "@DrBecky",
    "@acollierastro",
    "@zackdfilms",
    "@PrimerBlobs",
    "@BobbyBroccoli",
]

CHANNELS_TARGET = 40
VIDEOS_PER_CHANNEL = 120

OUTPUT_CSV = os.path.join("scripts", f"{CATEGORY_NAME}_channels_videos.csv")

# ============================================================
# SYSTEM CONFIG (do not edit)
# ============================================================

try:
    # Optional; if not installed, script still works via env var.
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY env var. Example: export YOUTUBE_API_KEY='...'")

# Thumbnail keys that YouTube may provide in snippet.thumbnails
THUMB_KEYS = ["default", "medium", "high", "standard", "maxres"]


# -----------------------------
# Helpers
# -----------------------------
def yt_client():
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def join_list(x: Optional[List[str]]) -> str:
    if not x:
        return ""
    return "|".join([str(i) for i in x])


def api_call_with_backoff(fn, max_retries: int = 7):
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


def extract_thumbnails(thumbnails: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns columns:
      thumb_default_url, thumb_default_width, thumb_default_height, ...
      for all keys in THUMB_KEYS (blank if not present).
    """
    out: Dict[str, Any] = {}
    if not isinstance(thumbnails, dict):
        thumbnails = {}

    for k in THUMB_KEYS:
        v = thumbnails.get(k, {}) if isinstance(thumbnails.get(k, {}), dict) else {}
        out[f"thumb_{k}_url"] = v.get("url", "")
        out[f"thumb_{k}_width"] = v.get("width", "")
        out[f"thumb_{k}_height"] = v.get("height", "")
    return out


# -----------------------------
# Resolve handle -> channelId (search.list)
# -----------------------------
def resolve_channel_id(youtube, handle_or_query: str) -> str:
    q = handle_or_query.strip()
    req = youtube.search().list(part="snippet", q=q, type="channel", maxResults=1)
    resp = api_call_with_backoff(req.execute)
    items = resp.get("items", [])

    if not items and q.startswith("@"):
        q2 = q[1:]
        req2 = youtube.search().list(part="snippet", q=q2, type="channel", maxResults=1)
        resp2 = api_call_with_backoff(req2.execute)
        items = resp2.get("items", [])

    if not items:
        raise RuntimeError(f"No channel found for: {handle_or_query}")

    channel_id = safe_get(items[0], ["snippet", "channelId"])
    if not channel_id:
        raise RuntimeError(f"Search returned item without channelId for: {handle_or_query}")
    return channel_id


# -----------------------------
# Channel + videos
# -----------------------------
def fetch_channel_details(youtube, channel_id: str) -> Dict[str, Any]:
    req = youtube.channels().list(
        part="snippet,contentDetails,statistics,brandingSettings,status,topicDetails",
        id=channel_id,
        maxResults=1,
    )
    resp = api_call_with_backoff(req.execute)
    items = resp.get("items", [])
    if not items:
        raise RuntimeError(f"No channel details returned for channelId: {channel_id}")
    return items[0]


def fetch_upload_video_ids(youtube, uploads_playlist_id: str, max_videos: int) -> List[str]:
    video_ids: List[str] = []
    page_token: Optional[str] = None

    while len(video_ids) < max_videos:
        req = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=min(50, max_videos - len(video_ids)),
            pageToken=page_token,
        )
        resp = api_call_with_backoff(req.execute)

        for it in resp.get("items", []):
            vid = safe_get(it, ["contentDetails", "videoId"])
            if vid:
                video_ids.append(vid)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return video_ids


def fetch_videos_details(youtube, video_ids: List[str]) -> List[Dict[str, Any]]:
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
        resp = api_call_with_backoff(req.execute)
        out.extend(resp.get("items", []))
    return out


# -----------------------------
# Row builders (CSV schema)
# -----------------------------
def channel_fields(channel: Dict[str, Any], handle: str) -> Dict[str, Any]:
    snippet = channel.get("snippet", {}) or {}
    stats = channel.get("statistics", {}) or {}
    branding = safe_get(channel, ["brandingSettings", "channel"], {}) or {}
    status = channel.get("status", {}) or {}
    topic = channel.get("topicDetails", {}) or {}

    uploads_pid = safe_get(channel, ["contentDetails", "relatedPlaylists", "uploads"], "")

    return {
        "snapshot_utc": iso_now(),
        "category_name": CATEGORY_NAME,

        "channel_handle_used": handle,
        "channel_id": channel.get("id", ""),
        "channel_title": snippet.get("title", ""),
        "channel_description": snippet.get("description", ""),
        "channel_publishedAt": snippet.get("publishedAt", ""),

        # Stored for traceability + incremental refresh later
        "uploads_playlist_id": uploads_pid,

        "channel_country": branding.get("country", ""),
        "channel_keywords": branding.get("keywords", ""),
        "channel_defaultLanguage": branding.get("defaultLanguage", ""),

        "channel_madeForKids": status.get("madeForKids", ""),
        "channel_isLinked": status.get("isLinked", ""),

        "channel_subscriberCount": stats.get("subscriberCount", ""),
        "channel_viewCount": stats.get("viewCount", ""),
        "channel_videoCount": stats.get("videoCount", ""),

        "channel_topicCategories": join_list(topic.get("topicCategories")),
        "channel_topicIds": join_list(topic.get("topicIds")),
    }


def video_row(video: Dict[str, Any], ch: Dict[str, Any]) -> Dict[str, Any]:
    sn = video.get("snippet", {}) or {}
    cd = video.get("contentDetails", {}) or {}
    st = video.get("statistics", {}) or {}
    vs = video.get("status", {}) or {}
    tp = video.get("topicDetails", {}) or {}

    thumbs = sn.get("thumbnails", {}) or {}
    thumb_cols = extract_thumbnails(thumbs)

    return {
        **ch,

        "video_id": video.get("id", ""),
        "video_title": sn.get("title", ""),
        "video_description": sn.get("description", ""),
        "video_publishedAt": sn.get("publishedAt", ""),
        "video_channelId": sn.get("channelId", ""),

        "video_categoryId": sn.get("categoryId", ""),
        "video_tags": join_list(sn.get("tags")),
        "video_defaultLanguage": sn.get("defaultLanguage", ""),
        "video_defaultAudioLanguage": sn.get("defaultAudioLanguage", ""),

        # all thumbnails (urls + dims)
        **thumb_cols,

        "views": st.get("viewCount", ""),
        "likes": st.get("likeCount", ""),
        "comments": st.get("commentCount", ""),

        "duration": cd.get("duration", ""),
        "caption": cd.get("caption", ""),
        "licensedContent": cd.get("licensedContent", ""),
        "definition": cd.get("definition", ""),
        "projection": cd.get("projection", ""),

        "madeForKids": vs.get("madeForKids", ""),
        "embeddable": vs.get("embeddable", ""),

        "video_topicCategories": join_list(tp.get("topicCategories")),
        "video_topicIds": join_list(tp.get("topicIds")),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    youtube = yt_client()

    handles = [h.strip() for h in CHANNEL_HANDLES if str(h).strip()]
    if len(handles) < CHANNELS_TARGET:
        print(f"Note: CHANNEL_HANDLES has {len(handles)} channels; team target is {CHANNELS_TARGET}. Running with provided list.")
    elif len(handles) > CHANNELS_TARGET:
        handles = handles[:CHANNELS_TARGET]
        print(f"Note: Trimming CHANNEL_HANDLES to first {CHANNELS_TARGET} channels.")

    rows: List[Dict[str, Any]] = []

    for handle in handles:
        print(f"\n=== Channel: {handle} ===")

        # Resolve channel ID (skip on failure; do not write error rows)
        try:
            channel_id = resolve_channel_id(youtube, handle)
        except Exception as e:
            print(f"SKIP: failed to resolve {handle}: {e}")
            continue

        # Fetch channel details
        try:
            channel = fetch_channel_details(youtube, channel_id)
        except Exception as e:
            print(f"SKIP: failed channel details for {handle} ({channel_id}): {e}")
            continue

        uploads_pid = safe_get(channel, ["contentDetails", "relatedPlaylists", "uploads"], "")
        if not uploads_pid:
            print(f"SKIP: missing uploads playlist for {handle} ({channel_id})")
            continue

        ch = channel_fields(channel, handle)

        # Fetch up to VIDEOS_PER_CHANNEL
        try:
            video_ids = fetch_upload_video_ids(youtube, uploads_pid, VIDEOS_PER_CHANNEL)
        except Exception as e:
            print(f"SKIP: failed uploads listing for {handle}: {e}")
            continue

        if not video_ids:
            print(f"SKIP: no video IDs found for {handle}")
            continue

        # Fetch video details (may return fewer if private/deleted)
        try:
            videos = fetch_videos_details(youtube, video_ids)
        except Exception as e:
            print(f"SKIP: failed video details for {handle}: {e}")
            continue

        if not videos:
            print(f"SKIP: no accessible video details returned for {handle}")
            continue

        print(f"Collected IDs: {len(video_ids)} (requested {VIDEOS_PER_CHANNEL})")
        print(f"Fetched video details: {len(videos)}")

        for v in videos:
            if not isinstance(v.get("snippet", {}), dict) or not v.get("id"):
                continue
            rows.append(video_row(v, ch))

    if not rows:
        raise RuntimeError("No data collected. CSV not written. Check handles, API key, and quota.")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if os.path.exists(OUTPUT_CSV):
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        print("Appended to existing CSV.")
    else:
        df.to_csv(OUTPUT_CSV, index=False)
        print("Created new CSV.")


    print(f"\nWrote dataset: {OUTPUT_CSV}")
    print(f"Rows: {len(df)}  Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
