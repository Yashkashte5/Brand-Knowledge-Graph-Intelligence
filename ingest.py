import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from apify_client import ApifyClient

load_dotenv()

DATA_DIR       = Path("data")
RAW_PATH       = DATA_DIR / "nike_raw.json"
PROCESSED_PATH = DATA_DIR / "nike_processed.json"
EMBED_PATH     = DATA_DIR / "graph_store" / "embeddings.pkl"

APIFY_TOKEN = os.environ["APIFY_API_TOKEN"]
ACTOR_ID    = "shu8hvrXbJbY3Eb9W"
NIKE_URL    = "https://www.instagram.com/nike/"
DAYS_BACK   = 60
MAX_POSTS   = 200

CUTOFF = datetime.now(tz=timezone.utc) - timedelta(days=DAYS_BACK)

THEME_MAP = {
    "olympics":       ["olympics","olympic","milanocortina2026","paris2024","la2028","athlete","gold","medal"],
    "running":        ["run","running","runner","marathon","5k","10k","pace","sprint","track","road"],
    "basketball":     ["basketball","nba","court","hoop","jordan","airjordan","lebron","kobe"],
    "training":       ["train","training","workout","gym","fitness","strength","muscle","lift","sweat"],
    "football":       ["football","soccer","fifa","worldcup","pitch","goal","cleats"],
    "sustainability": ["sustainable","sustainability","planet","green","recycle","movetozero","forward"],
    "fashion":        ["style","fashion","streetwear","drip","outfit","fit","look","wear"],
    "women":          ["women","woman","girl","she","her","female","nikewoman","nikewomen"],
    "kids":           ["kids","child","children","future","youth","junior","play"],
    "just_do_it":     ["justdoit","justdo","motivation","inspire","inspiration","believe","dream"],
    "air_max":        ["airmax","airforce","af1","airforce1","sneaker","sneakerhead","kicks","shoe"],
}

def fetch_raw_posts() -> list[dict]:
    client = ApifyClient(APIFY_TOKEN)
    run_input = {
        "directUrls":    [NIKE_URL],
        "resultsType":   "posts",
        "resultsLimit":  MAX_POSTS,
        "addParentData": False,
    }
    print(f"[ingest] Starting Apify actor {ACTOR_ID} for @nike ...")
    run   = client.actor(ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"[ingest] Received {len(items)} raw items.")
    return items

def _parse_ts(raw) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None

def _caption(item: dict) -> str:
    cap = item.get("caption") or item.get("edge_media_to_caption", {})
    if isinstance(cap, str):
        return cap
    if isinstance(cap, dict):
        edges = cap.get("edges", [])
        return edges[0].get("node", {}).get("text", "") if edges else ""
    return ""

def _media_type(item: dict) -> str:
    raw = item.get("type") or item.get("media_type") or "image"
    if isinstance(raw, int):
        return {1: "image", 2: "video", 8: "carousel"}.get(raw, "image")
    m = {
        "image": "image", "graphimage": "image",
        "video": "video", "graphvideo": "video",
        "sidecar": "carousel", "graphsidecar": "carousel", "carousel": "carousel",
    }
    return m.get(raw.lower(), "image")

def _post_id(item: dict) -> str:
    return str(
        item.get("shortCode") or item.get("shortcode")
        or item.get("id") or item.get("pk") or ""
    )

def _extract_hashtags(caption: str, apify_tags: list) -> list[str]:
    tags = set()
    for h in (apify_tags or []):
        if isinstance(h, str):
            tags.add(h.lstrip("#").lower())
    for t in re.findall(r"#(\w+)", caption):
        tags.add(t.lower())
    return sorted(tags)

def _extract_mentions(caption: str, apify_mentions: list) -> list[str]:
    mentions = set()
    for m in (apify_mentions or []):
        if isinstance(m, str):
            mentions.add(m.lstrip("@").lower())
    for m in re.findall(r"@(\w+)", caption):
        if m.lower() != "nike":
            mentions.add(m.lower())
    return sorted(mentions)

def _extract_themes(caption: str, hashtags: list[str]) -> list[str]:
    combined = caption.lower() + " " + " ".join(hashtags)
    return [
        theme for theme, keywords in THEME_MAP.items()
        if any(kw in combined for kw in keywords)
    ]

def build_embeddings(posts: list[dict]) -> None:
    """Build and cache caption embeddings using sentence-transformers (local, free)."""
    try:
        from sentence_transformers import SentenceTransformer
        import pickle
    except ImportError:
        print("[ingest] sentence-transformers not installed — skipping embeddings.")
        print("[ingest] Run: pip install sentence-transformers")
        return

    print("[ingest] Building caption embeddings (all-MiniLM-L6-v2) ...")
    model    = SentenceTransformer("all-MiniLM-L6-v2")
    ids      = [p["post_id"] for p in posts]
    captions = [p["caption"] or p["post_id"] for p in posts]
    vectors  = model.encode(captions, show_progress_bar=True, convert_to_numpy=True)

    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_PATH, "wb") as f:
        pickle.dump({"ids": ids, "vectors": vectors}, f)
    print(f"[ingest] Embeddings saved -> {EMBED_PATH}  ({len(ids)} vectors)")

def normalise(item: dict) -> Optional[dict]:
    ts = _parse_ts(item.get("timestamp") or item.get("taken_at_timestamp"))
    if ts is None or ts < CUTOFF:
        return None
    pid = _post_id(item)
    if not pid:
        return None

    caption  = _caption(item)
    hashtags = _extract_hashtags(caption, item.get("hashtags") or [])
    mentions = _extract_mentions(caption, item.get("mentions") or [])
    themes   = _extract_themes(caption, hashtags)

    return {
        "post_id":       pid,
        "url":           item.get("url") or f"https://www.instagram.com/p/{pid}/",
        "caption":       caption,
        "hashtags":      hashtags,
        "mentions":      mentions,
        "themes":        themes,
        "like_count":    int(item.get("likesCount")    or item.get("edge_media_preview_like", {}).get("count", 0) or 0),
        "comment_count": int(item.get("commentsCount") or item.get("edge_media_to_comment",  {}).get("count", 0) or 0),
        "timestamp":     ts.isoformat(),
        "month":         ts.strftime("%Y-%m"),
        "media_type":    _media_type(item),
    }

def run_ingestion() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_items = fetch_raw_posts()
    RAW_PATH.write_text(json.dumps(raw_items, indent=2, default=str))
    print(f"[ingest] Raw  -> {RAW_PATH}  ({len(raw_items)} items)")

    processed, seen = [], set()
    for item in raw_items:
        p = normalise(item)
        if p and p["post_id"] not in seen:
            seen.add(p["post_id"])
            processed.append(p)

    PROCESSED_PATH.write_text(json.dumps(processed, indent=2))

    total_tags     = sum(len(p["hashtags"]) for p in processed)
    total_mentions = sum(len(p["mentions"]) for p in processed)
    total_themes   = sum(len(p["themes"])   for p in processed)
    print(f"[ingest] Processed -> {PROCESSED_PATH}  ({len(processed)} posts in last {DAYS_BACK} days)")
    print(f"[ingest] Extracted: {total_tags} hashtags  {total_mentions} mentions  {total_themes} theme tags")

    # Always rebuild embeddings after fresh ingestion
    build_embeddings(processed)

    return processed

if __name__ == "__main__":
    run_ingestion()