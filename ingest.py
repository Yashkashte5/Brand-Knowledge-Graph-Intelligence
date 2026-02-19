from __future__ import annotations

import json
import os
import pickle
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from apify_client import ApifyClient
from dotenv import load_dotenv
from keybert import KeyBERT
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

load_dotenv()

DATA_DIR  = Path("data")
ACTOR_ID  = "shu8hvrXbJbY3Eb9W"
DAYS_BACK = 60
MAX_POSTS  = 200

EMBEDDING_MODEL     = "BAAI/bge-small-en-v1.5"
BGE_DOCUMENT_PREFIX = "Represent this sentence: "
BGE_QUERY_PREFIX    = "Represent this sentence: "

N_CLUSTERS      = 8     # number of themes to discover per account
MIN_THEME_SCORE = 0.20  # min cosine similarity to cluster centre to be assigned a theme

STOP_WORDS = set(stopwords.words("english"))

_embedding_model = SentenceTransformer(EMBEDDING_MODEL)
_keybert_model   = KeyBERT(model=_embedding_model)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _account_dir(username: str) -> Path:
    return DATA_DIR / username.lower().lstrip("@")

def raw_path(username: str) -> Path:
    return _account_dir(username) / "raw.json"

def processed_path(username: str) -> Path:
    return _account_dir(username) / "processed.json"

def embed_path(username: str) -> Path:
    return _account_dir(username) / "graph_store" / "embeddings.pkl"

def graph_path(username: str) -> Path:
    return _account_dir(username) / "graph_store" / "graph.gpickle"

def graph_html_path(username: str, suffix: str = "") -> Path:
    return _account_dir(username) / f"graph{suffix}.html"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

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
    return {
        "image": "image", "graphimage": "image",
        "video": "video", "graphvideo": "video",
        "sidecar": "carousel", "graphsidecar": "carousel", "carousel": "carousel",
    }.get(str(raw).lower(), "image")

def _post_id(item: dict) -> str:
    return str(item.get("shortCode") or item.get("shortcode") or item.get("id") or item.get("pk") or "")

def _extract_hashtags(caption: str, apify_tags: list) -> list[str]:
    tags = set()
    for h in (apify_tags or []):
        if isinstance(h, str):
            tags.add(h.lstrip("#").lower())
    for t in re.findall(r"#(\w+)", caption):
        tags.add(t.lower())
    return sorted(tags)

def _extract_mentions(caption: str, apify_mentions: list, username: str) -> list[str]:
    mentions = set()
    for m in (apify_mentions or []):
        if isinstance(m, str):
            mentions.add(m.lstrip("@").lower())
    for m in re.findall(r"@(\w+)", caption):
        if m.lower() != username.lower():
            mentions.add(m.lower())
    return sorted(mentions)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_themes(posts: list[dict]) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """
    Cluster post captions using BGE embeddings + KMeans.
    Label each cluster with KeyBERT keywords.

    Returns:
        post_themes       — {post_id: [theme_label, ...]}
        post_theme_scores — {post_id: {theme_label: score}}
    """
    captioned = [p for p in posts if (p.get("caption") or "").strip()]
    if len(captioned) < 2:
        return {p["post_id"]: [] for p in posts}, {}

    texts   = [BGE_DOCUMENT_PREFIX + p["caption"] for p in captioned]
    vectors = _embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    n_clusters = min(N_CLUSTERS, len(captioned))
    km      = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels  = km.fit_predict(vectors)
    centers = km.cluster_centers_

    cluster_texts: dict[int, list[str]] = {}
    for i, p in enumerate(captioned):
        cluster_texts.setdefault(int(labels[i]), []).append(p["caption"])

    cluster_labels: dict[int, str] = {}
    for cid, ctexts in cluster_texts.items():
        combined = " ".join(ctexts)

        kws = _keybert_model.extract_keywords(
            combined,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=2,
            use_mmr=True,
            diversity=0.5,
        )
        if kws:
            cluster_labels[cid] = " & ".join(kw for kw, _ in kws)
        else:
            words = re.findall(r"[a-zA-Z]{3,}", combined.lower())
            top   = [w for w, _ in Counter(w for w in words if w not in STOP_WORDS).most_common(3)]
            cluster_labels[cid] = " & ".join(top) if top else f"theme_{cid}"

    post_themes:       dict[str, list[str]]        = {}
    post_theme_scores: dict[str, dict[str, float]] = {}

    for i, p in enumerate(captioned):
        pid        = p["post_id"]
        cluster_id = int(labels[i])
        center     = centers[cluster_id]
        score      = float(np.dot(vectors[i], center / (np.linalg.norm(center) + 1e-10)))
        theme      = cluster_labels[cluster_id]
        post_themes[pid]       = [theme] if score >= MIN_THEME_SCORE else []
        post_theme_scores[pid] = {theme: round(score, 4)}

    for p in posts:
        if p["post_id"] not in post_themes:
            post_themes[p["post_id"]]       = []
            post_theme_scores[p["post_id"]] = {}

    return post_themes, post_theme_scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalise(item: dict, username: str) -> Optional[dict]:
    """Normalise a raw Apify item — themes assigned later via clustering."""
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=DAYS_BACK)
    ts = _parse_ts(item.get("timestamp") or item.get("taken_at_timestamp"))
    if ts is None or ts < cutoff:
        return None
    pid = _post_id(item)
    if not pid:
        return None
    caption  = _caption(item)
    hashtags = _extract_hashtags(caption, item.get("hashtags") or [])
    mentions = _extract_mentions(caption, item.get("mentions") or [], username)
    return {
        "post_id":       pid,
        "username":      username.lower().lstrip("@"),
        "url":           item.get("url") or f"https://www.instagram.com/p/{pid}/",
        "caption":       caption,
        "hashtags":      hashtags,
        "mentions":      mentions,
        "themes":        [],
        "theme_scores":  {},
        "like_count":    int(item.get("likesCount")    or item.get("edge_media_preview_like", {}).get("count", 0) or 0),
        "comment_count": int(item.get("commentsCount") or item.get("edge_media_to_comment",  {}).get("count", 0) or 0),
        "timestamp":     ts.isoformat(),
        "month":         ts.strftime("%Y-%m"),
        "media_type":    _media_type(item),
    }


def build_embeddings(posts: list[dict], username: str) -> None:
    """Encode post captions with BAAI/bge-small-en-v1.5 and persist to disk."""
    ep = embed_path(username)
    ep.parent.mkdir(parents=True, exist_ok=True)

    ids     = [p["post_id"] for p in posts]
    caps    = [BGE_DOCUMENT_PREFIX + (p["caption"] or p["post_id"]) for p in posts]
    vectors = _embedding_model.encode(caps, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    with open(ep, "wb") as f:
        pickle.dump({"ids": ids, "vectors": vectors, "model": EMBEDDING_MODEL}, f)


def scrape_account(username: str, force: bool = False) -> list[dict]:
    username = username.lower().lstrip("@")
    pp = processed_path(username)
    rp = raw_path(username)

    if not force and pp.exists():
        try:
            posts = json.loads(pp.read_text())
            if posts:
                return posts
        except json.JSONDecodeError:
            pass

    _account_dir(username).mkdir(parents=True, exist_ok=True)

    if not force and rp.exists():
        try:
            items = json.loads(rp.read_text())
        except json.JSONDecodeError:
            items = None
    else:
        items = None

    if items is None:
        apify_token = os.environ.get("APIFY_API_TOKEN")
        if not apify_token:
            raise EnvironmentError("APIFY_API_TOKEN not set in .env")
        client = ApifyClient(apify_token)
        run = client.actor(ACTOR_ID).call(run_input={
            "directUrls":    [f"https://www.instagram.com/{username}/"],
            "resultsType":   "posts",
            "resultsLimit":  MAX_POSTS,
            "addParentData": False,
        })
        items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        rp.write_text(json.dumps(items, indent=2, default=str))

    processed, seen = [], set()
    for item in items:
        p = normalise(item, username)
        if p and p["post_id"] not in seen:
            seen.add(p["post_id"])
            processed.append(p)

    print(f"Clustering {len(processed)} posts into themes...")
    post_themes, post_theme_scores = _cluster_themes(processed)
    for p in processed:
        p["themes"]       = post_themes.get(p["post_id"], [])
        p["theme_scores"] = post_theme_scores.get(p["post_id"], {})

    pp.write_text(json.dumps(processed, indent=2))
    build_embeddings(processed, username)
    return processed


def get_cached_accounts() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return [d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / "processed.json").exists()]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <instagram_username>")
        sys.exit(1)
    scrape_account(sys.argv[1])