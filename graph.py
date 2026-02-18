"""
graph.py — Nike Instagram Knowledge Graph Engine
Pure networkx graph — zero LLM / OpenAI dependency.
Similarity search uses sentence-transformers (local, free).
All analytics are deterministic Python.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR       = Path("data")
PROCESSED_PATH = DATA_DIR / "nike_processed.json"
GRAPH_PATH     = DATA_DIR / "graph_store" / "graph.gpickle"
EMBED_PATH     = DATA_DIR / "graph_store" / "embeddings.pkl"

# ---------------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------------

def build_graph(force_rebuild: bool = False) -> nx.DiGraph:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and GRAPH_PATH.exists():
        try:
            return _load()
        except Exception as e:
            print(f"[graph] ️  Existing graph unreadable ({e}) — rebuilding …")
            GRAPH_PATH.unlink(missing_ok=True)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Run ingest.py first — {PROCESSED_PATH} not found.")

    posts: list[dict] = json.loads(PROCESSED_PATH.read_text())
    print(f"[graph] Building graph from {len(posts)} posts …")

    G = nx.DiGraph()

    # Brand node
    G.add_node("Nike", node_type="brand", label="Nike")

    for p in posts:
        pid = p["post_id"]

        # Post node
        G.add_node(pid,
                   node_type="post",
                   label=f"Post:{pid[:8]}",
                   like_count=p["like_count"],
                   comment_count=p["comment_count"],
                   media_type=p["media_type"],
                   timestamp=p["timestamp"],
                   month=p["month"],
                   caption=p["caption"],
                   url=p["url"])

        G.add_edge("Nike", pid, relation="POSTED")

        # Hashtag nodes
        for tag in p.get("hashtags", []):
            tag_id = f"#{tag}"
            if not G.has_node(tag_id):
                G.add_node(tag_id, node_type="hashtag", label=tag_id)
            G.add_edge(pid, tag_id, relation="HAS_HASHTAG")

        # Mention nodes (@athlete / collaborator)
        for mention in p.get("mentions", []):
            m_id = f"@{mention}"
            if not G.has_node(m_id):
                G.add_node(m_id, node_type="mention", label=m_id)
            G.add_edge(pid, m_id, relation="MENTIONS")

        # Theme nodes (campaign themes inferred from caption)
        for theme in p.get("themes", []):
            t_id = f"theme:{theme}"
            if not G.has_node(t_id):
                G.add_node(t_id, node_type="theme", label=theme.replace("_", " ").title())
            G.add_edge(pid, t_id, relation="THEME")

        # Month node
        month_id = f"month:{p['month']}"
        if not G.has_node(month_id):
            G.add_node(month_id, node_type="month", label=p["month"])
        G.add_edge(pid, month_id, relation="BELONGS_TO")

        # MediaType node
        mt_id = f"media:{p['media_type']}"
        if not G.has_node(mt_id):
            G.add_node(mt_id, node_type="media_type", label=p["media_type"].upper())
        G.add_edge(pid, mt_id, relation="IS_TYPE")

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)

    print(f"[graph] Graph saved -> {GRAPH_PATH}  "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    return G


def _load() -> nx.DiGraph:
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Invalid graph file")
    return G


def load_graph() -> nx.DiGraph:
    if GRAPH_PATH.exists():
        return _load()
    return build_graph()


# ---------------------------------------------------------------------------
# Embeddings for similarity search (sentence-transformers, local)
# ---------------------------------------------------------------------------

def _build_embeddings(posts: list[dict]) -> dict[str, Any]:
    """Build and cache caption embeddings using a local sentence-transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return {}

    print("[graph] Building caption embeddings (sentence-transformers) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80 MB, downloads once

    ids      = [p["post_id"] for p in posts]
    captions = [p["caption"] or p["post_id"] for p in posts]
    vectors  = model.encode(captions, show_progress_bar=True)

    data = {"ids": ids, "vectors": vectors}
    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"[graph] Embeddings saved -> {EMBED_PATH}")
    return data


def _load_embeddings() -> dict:
    if not EMBED_PATH.exists():
        return {}
    with open(EMBED_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Data loader helper
# ---------------------------------------------------------------------------

def _posts() -> list[dict]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("Run ingest.py first.")
    return json.loads(PROCESSED_PATH.read_text())


# ---------------------------------------------------------------------------
# Analytics (all deterministic, no LLM)
# ---------------------------------------------------------------------------

def get_engagement_summary() -> dict[str, Any]:
    posts = _posts()
    if not posts:
        return {"error": "No posts found."}

    likes    = [p["like_count"]    for p in posts]
    comments = [p["comment_count"] for p in posts]
    s_likes  = sorted(likes)
    n        = len(s_likes)
    median   = s_likes[n // 2] if n % 2 else (s_likes[n//2-1] + s_likes[n//2]) // 2
    best     = max(posts, key=lambda p: p["like_count"])

    media: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    monthly: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    for p in posts:
        media[p["media_type"]]["count"]    += 1
        media[p["media_type"]]["likes"]    += p["like_count"]
        media[p["media_type"]]["comments"] += p["comment_count"]
        monthly[p["month"]]["count"]       += 1
        monthly[p["month"]]["likes"]       += p["like_count"]
        monthly[p["month"]]["comments"]    += p["comment_count"]

    media_breakdown = {
        mt: {
            "post_count":   v["count"],
            "avg_likes":    v["likes"] // v["count"],
            "avg_comments": v["comments"] // v["count"],
        }
        for mt, v in media.items()
    }
    monthly_trend = {
        m: {
            "post_count":   v["count"],
            "avg_likes":    v["likes"] // v["count"],
            "avg_comments": v["comments"] // v["count"],
        }
        for m, v in sorted(monthly.items())
    }

    return {
        "total_posts":   len(posts),
        "total_likes":   sum(likes),
        "total_comments":sum(comments),
        "avg_likes":     sum(likes) // len(likes),
        "median_likes":  median,
        "avg_comments":  sum(comments) // len(comments),
        "best_post": {
            "post_id":        best["post_id"],
            "url":            best["url"],
            "like_count":     best["like_count"],
            "comment_count":  best["comment_count"],
            "media_type":     best["media_type"],
            "caption_preview":(best["caption"] or "")[:140],
        },
        "media_breakdown": media_breakdown,
        "monthly_trend":   monthly_trend,
    }


def get_hashtag_analysis() -> dict[str, Any]:
    posts = _posts()
    stats: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})

    for p in posts:
        for tag in p["hashtags"]:
            stats[tag]["count"]    += 1
            stats[tag]["likes"]    += p["like_count"]
            stats[tag]["comments"] += p["comment_count"]

    rows = [
        {
            "hashtag":      f"#{tag}",
            "frequency":    v["count"],
            "avg_likes":    v["likes"] // v["count"],
            "avg_comments": v["comments"] // v["count"],
        }
        for tag, v in stats.items()
    ]
    rows.sort(key=lambda x: x["frequency"], reverse=True)

    return {
        "total_unique_hashtags": len(rows),
        "top_by_frequency":      rows[:25],
        "top_by_avg_likes":      sorted(rows, key=lambda x: x["avg_likes"], reverse=True)[:10],
    }


def get_monthly_breakdown() -> dict[str, Any]:
    posts = _posts()
    monthly: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0, "media": defaultdict(int)})
    for p in posts:
        m = p["month"]
        monthly[m]["count"]    += 1
        monthly[m]["likes"]    += p["like_count"]
        monthly[m]["comments"] += p["comment_count"]
        monthly[m]["media"][p["media_type"]] += 1

    return {
        m: {
            "posts":        v["count"],
            "total_likes":  v["likes"],
            "avg_likes":    v["likes"] // v["count"],
            "avg_comments": v["comments"] // v["count"],
            "media_mix":    dict(v["media"]),
        }
        for m, v in sorted(monthly.items())
    }


def find_similar_posts(post_id: str, top_k: int = 5) -> dict[str, Any]:
    """
    Similarity search using:
    1. sentence-transformers caption embeddings (if available)
    2. Hashtag Jaccard similarity (fallback)
    """
    posts    = _posts()
    post_map = {p["post_id"]: p for p in posts}

    if post_id not in post_map:
        return {"error": f"post_id {post_id!r} not found."}

    target = post_map[post_id]

    # Try embedding similarity first
    embed_data = _load_embeddings()
    if embed_data and "ids" in embed_data:
        import numpy as np
        ids     = embed_data["ids"]
        vectors = embed_data["vectors"]
        if post_id in ids:
            idx    = ids.index(post_id)
            q_vec  = vectors[idx]
            norms  = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec)
            norms  = np.where(norms == 0, 1e-10, norms)
            scores = (vectors @ q_vec) / norms
            ranked = sorted(
                [(ids[i], float(scores[i])) for i in range(len(ids)) if ids[i] != post_id],
                key=lambda x: x[1], reverse=True
            )[:top_k]

            return {
                "query_post_id": post_id,
                "method": "sentence_transformer_cosine",
                "similar_posts": [
                    {
                        "post_id":        pid,
                        "similarity":     round(score, 4),
                        "like_count":     post_map[pid]["like_count"],
                        "media_type":     post_map[pid]["media_type"],
                        "caption_preview":(post_map[pid]["caption"] or "")[:80],
                    }
                    for pid, score in ranked if pid in post_map
                ],
            }

    # Fallback: Jaccard on hashtags
    target_tags = set(target["hashtags"])
    sims = []
    for p in posts:
        if p["post_id"] == post_id:
            continue
        p_tags = set(p["hashtags"])
        union  = target_tags | p_tags
        inter  = target_tags & p_tags
        j      = len(inter) / len(union) if union else 0.0
        sims.append({
            "post_id":        p["post_id"],
            "similarity":     round(j, 4),
            "shared_hashtags":list(inter),
            "like_count":     p["like_count"],
            "media_type":     p["media_type"],
            "caption_preview":(p["caption"] or "")[:80],
        })
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "query_post_id": post_id,
        "method": "hashtag_jaccard",
        "similar_posts": sims[:top_k],
    }


def extract_subgraph(filters: Optional[dict] = None) -> dict[str, Any]:
    """Return node/edge lists for visualisation, with optional filters."""
    posts = _posts()

    if filters:
        if "month"      in filters: posts = [p for p in posts if p["month"]      == filters["month"]]
        if "media_type" in filters: posts = [p for p in posts if p["media_type"] == filters["media_type"]]
        if "min_likes"  in filters: posts = [p for p in posts if p["like_count"] >= filters["min_likes"]]

    nodes: list[dict] = []
    edges: list[dict] = []
    seen: dict = {}   # id -> True, to deduplicate all non-post nodes

    max_likes = max((p["like_count"] for p in posts), default=1)

    nodes.append({"id": "Nike", "label": "Nike", "group": "brand", "size": 50})

    for p in posts:
        pid  = p["post_id"]
        size = int(12 + 30 * (p["like_count"] / max(max_likes, 1)))

        nodes.append({
            "id":           pid,
            "label":        f"Post\n{pid[:8]}",
            "group":        "post",
            "size":         size,
            "like_count":   p["like_count"],
            "comment_count":p["comment_count"],
            "media_type":   p["media_type"],
            "url":          p["url"],
            "caption":      (p.get("caption") or "")[:120],
        })
        edges.append({"from": "Nike", "to": pid, "label": "POSTED"})

        for tag in p.get("hashtags", []):
            tid = f"#{tag}"
            if tid not in seen:
                nodes.append({"id": tid, "label": tid, "group": "hashtag", "size": 14})
                seen[tid] = True
            edges.append({"from": pid, "to": tid, "label": "HAS_HASHTAG"})

        for mention in p.get("mentions", []):
            mid = f"@{mention}"
            if mid not in seen:
                nodes.append({"id": mid, "label": mid, "group": "mention", "size": 16})
                seen[mid] = True
            edges.append({"from": pid, "to": mid, "label": "MENTIONS"})

        for theme in p.get("themes", []):
            thid = f"theme:{theme}"
            if thid not in seen:
                label = theme.replace("_", " ").title()
                nodes.append({"id": thid, "label": label, "group": "theme", "size": 22})
                seen[thid] = True
            edges.append({"from": pid, "to": thid, "label": "THEME"})

        month_id = f"month:{p['month']}"
        if month_id not in seen:
            nodes.append({"id": month_id, "label": p["month"], "group": "month", "size": 24})
            seen[month_id] = True
        edges.append({"from": pid, "to": month_id, "label": "BELONGS_TO"})

        mt_id = f"media:{p['media_type']}"
        if mt_id not in seen:
            nodes.append({"id": mt_id, "label": p["media_type"].upper(), "group": "media_type", "size": 20})
            seen[mt_id] = True
        edges.append({"from": pid, "to": mt_id, "label": "IS_TYPE"})

    total_tags     = sum(1 for k in seen if k.startswith("#"))
    total_mentions = sum(1 for k in seen if k.startswith("@"))
    total_themes   = sum(1 for k in seen if k.startswith("theme:"))

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "total_posts":    len(posts),
            "total_hashtags": total_tags,
            "total_mentions": total_mentions,
            "total_themes":   total_themes,
            "filters_applied":filters or {},
        },
    }


def get_graph_context() -> dict[str, Any]:
    """
    Return a rich structured snapshot of the entire graph for Claude to reason over.
    Claude (not this function) is responsible for interpreting and answering questions.
    """
    posts = _posts()
    if not posts:
        return {"error": "No posts found. Run ingest.py first."}

    return {
        "period":      "last 60 days",
        "brand":       "Nike (@nike)",
        "engagement":  get_engagement_summary(),
        "hashtags":    get_hashtag_analysis(),
        "monthly":     get_monthly_breakdown(),
        "posts": [
            {
                "post_id":      p["post_id"],
                "url":          p["url"],
                "caption":      p["caption"],
                "hashtags":     p["hashtags"],
                "like_count":   p["like_count"],
                "comment_count":p["comment_count"],
                "media_type":   p["media_type"],
                "month":        p["month"],
                "timestamp":    p["timestamp"],
            }
            for p in posts
        ],
    }