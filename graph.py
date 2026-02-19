from __future__ import annotations

import json
import pickle
from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np

from ingest import (
    processed_path, embed_path, graph_path, build_embeddings,
    DAYS_BACK, EMBEDDING_MODEL, BGE_QUERY_PREFIX,
)


def build_graph(username: str, force_rebuild: bool = False) -> nx.DiGraph:
    username = username.lower().lstrip("@")
    gp = graph_path(username)
    gp.parent.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and gp.exists():
        try:
            return _load(username)
        except Exception:
            gp.unlink(missing_ok=True)

    posts = _posts(username)
    G = nx.DiGraph()
    G.add_node(username, node_type="brand", label=f"@{username}")

    for p in posts:
        pid = p["post_id"]
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
        G.add_edge(username, pid, relation="POSTED")

        for tag in p.get("hashtags", []):
            tid = f"#{tag}"
            if not G.has_node(tid):
                G.add_node(tid, node_type="hashtag", label=tid)
            G.add_edge(pid, tid, relation="HAS_HASHTAG")

        for mention in p.get("mentions", []):
            mid = f"@{mention}"
            if not G.has_node(mid):
                G.add_node(mid, node_type="mention", label=mid)
            G.add_edge(pid, mid, relation="MENTIONS")

        for theme in p.get("themes", []):
            t_id = f"theme:{theme}"
            if not G.has_node(t_id):
                G.add_node(t_id, node_type="theme", label=theme.replace("_", " ").title())
            G.add_edge(pid, t_id, relation="THEME")

        month_id = f"month:{p['month']}"
        if not G.has_node(month_id):
            G.add_node(month_id, node_type="month", label=p["month"])
        G.add_edge(pid, month_id, relation="BELONGS_TO")

        mt_id = f"media:{p['media_type']}"
        if not G.has_node(mt_id):
            G.add_node(mt_id, node_type="media_type", label=p["media_type"].upper())
        G.add_edge(pid, mt_id, relation="IS_TYPE")

    with open(gp, "wb") as f:
        pickle.dump(G, f)
    return G


def _load(username: str) -> nx.DiGraph:
    with open(graph_path(username), "rb") as f:
        G = pickle.load(f)
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Invalid graph file")
    return G


def _posts(username: str) -> list[dict]:
    pp = processed_path(username)
    if not pp.exists():
        raise FileNotFoundError(f"No data for @{username}.")
    return json.loads(pp.read_text())


def _load_embeddings(username: str) -> dict:
    ep = embed_path(username)
    if not ep.exists():
        return {}
    with open(ep, "rb") as f:
        return pickle.load(f)


def _embeddings_are_fresh(username: str, posts: list[dict]) -> tuple[bool, dict]:
    """
    Return (is_fresh, embed_data).
    Stale = embed file missing, wrong model, or post IDs don't match current posts.
    """
    embed_data = _load_embeddings(username)
    if not embed_data or "ids" not in embed_data:
        return False, {}
    if embed_data.get("model") != EMBEDDING_MODEL:
        return False, {}
    if set(embed_data["ids"]) != {p["post_id"] for p in posts}:
        return False, {}
    return True, embed_data


def get_engagement_summary(username: str) -> dict[str, Any]:
    posts = _posts(username)
    if not posts:
        return {"error": "No posts found."}

    likes    = [p["like_count"]    for p in posts]
    comments = [p["comment_count"] for p in posts]
    best     = max(posts, key=lambda p: p["like_count"])

    media:   dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})
    monthly: dict = defaultdict(lambda: {"count": 0, "likes": 0, "comments": 0})

    for p in posts:
        media[p["media_type"]]["count"]    += 1
        media[p["media_type"]]["likes"]    += p["like_count"]
        media[p["media_type"]]["comments"] += p["comment_count"]
        monthly[p["month"]]["count"]       += 1
        monthly[p["month"]]["likes"]       += p["like_count"]
        monthly[p["month"]]["comments"]    += p["comment_count"]

    s = sorted(likes)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) // 2

    return {
        "username":       username,
        "total_posts":    len(posts),
        "total_likes":    sum(likes),
        "total_comments": sum(comments),
        "avg_likes":      sum(likes) // len(likes),
        "median_likes":   median,
        "avg_comments":   sum(comments) // len(comments),
        "best_post": {
            "post_id":        best["post_id"],
            "url":            best["url"],
            "like_count":     best["like_count"],
            "comment_count":  best["comment_count"],
            "media_type":     best["media_type"],
            "caption_preview":(best["caption"] or "")[:140],
        },
        "media_breakdown": {
            mt: {
                "post_count":   v["count"],
                "avg_likes":    v["likes"] // v["count"],
                "avg_comments": v["comments"] // v["count"],
            }
            for mt, v in media.items()
        },
        "monthly_trend": {
            m: {
                "post_count":   v["count"],
                "avg_likes":    v["likes"] // v["count"],
                "avg_comments": v["comments"] // v["count"],
            }
            for m, v in sorted(monthly.items())
        },
    }


def get_hashtag_analysis(username: str) -> dict[str, Any]:
    posts = _posts(username)
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
        "username":              username,
        "total_unique_hashtags": len(rows),
        "top_by_frequency":      rows[:25],
        "top_by_avg_likes":      sorted(rows, key=lambda x: x["avg_likes"], reverse=True)[:10],
    }


def get_monthly_breakdown(username: str) -> dict[str, Any]:
    posts = _posts(username)
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


def find_similar_posts(username: str, post_id: str, top_k: int = 5) -> dict[str, Any]:
    posts    = _posts(username)
    post_map = {p["post_id"]: p for p in posts}

    if post_id not in post_map:
        return {"error": f"post_id {post_id!r} not found in @{username}."}

    fresh, embed_data = _embeddings_are_fresh(username, posts)
    if not fresh:
        build_embeddings(posts, username)
        _, embed_data = _embeddings_are_fresh(username, posts)

    if embed_data and "ids" in embed_data:
        ids      = embed_data["ids"]
        vectors  = embed_data["vectors"]
        id_index = {pid: i for i, pid in enumerate(ids)}

        if post_id in id_index:
            idx    = id_index[post_id]
            scores = vectors @ vectors[idx]
            ranked = sorted(
                [(ids[i], float(scores[i])) for i in range(len(ids)) if ids[i] != post_id],
                key=lambda x: x[1], reverse=True
            )[:top_k]
            return {
                "query_post_id": post_id,
                "method":        "bge_cosine",
                "model":         EMBEDDING_MODEL,
                "similar_posts": [
                    {
                        "post_id":         pid,
                        "similarity":      round(score, 4),
                        "like_count":      post_map[pid]["like_count"],
                        "media_type":      post_map[pid]["media_type"],
                        "caption_preview": (post_map[pid]["caption"] or "")[:80],
                    }
                    for pid, score in ranked if pid in post_map
                ],
            }

    # Fallback: hashtag Jaccard similarity
    target_tags = set(post_map[post_id]["hashtags"])
    sims = []
    for p in posts:
        if p["post_id"] == post_id:
            continue
        p_tags = set(p["hashtags"])
        union  = target_tags | p_tags
        inter  = target_tags & p_tags
        sims.append({
            "post_id":         p["post_id"],
            "similarity":      round(len(inter) / len(union), 4) if union else 0.0,
            "shared_hashtags": list(inter),
            "like_count":      p["like_count"],
            "media_type":      p["media_type"],
            "caption_preview": (p["caption"] or "")[:80],
        })
    sims.sort(key=lambda x: x["similarity"], reverse=True)
    return {
        "query_post_id": post_id,
        "method":        "hashtag_jaccard_fallback",
        "warning":       "Embedding model unavailable; using hashtag overlap instead.",
        "similar_posts": sims[:top_k],
    }


def extract_subgraph(username: str, filters: Optional[dict] = None) -> dict[str, Any]:
    posts = _posts(username)
    if filters:
        if "month"      in filters: posts = [p for p in posts if p["month"]      == filters["month"]]
        if "media_type" in filters: posts = [p for p in posts if p["media_type"] == filters["media_type"]]
        if "min_likes"  in filters: posts = [p for p in posts if p["like_count"] >= filters["min_likes"]]

    nodes: list[dict] = []
    edges: list[dict] = []
    seen:  dict       = {}
    max_likes = max((p["like_count"] for p in posts), default=1)

    nodes.append({"id": username, "label": f"@{username}", "group": "brand", "size": 50})

    for p in posts:
        pid  = p["post_id"]
        size = int(12 + 30 * (p["like_count"] / max(max_likes, 1)))
        nodes.append({
            "id":            pid,
            "label":         f"Post\n{pid[:8]}",
            "group":         "post",
            "size":          size,
            "like_count":    p["like_count"],
            "comment_count": p["comment_count"],
            "media_type":    p["media_type"],
            "url":           p["url"],
            "caption":       (p.get("caption") or "")[:120],
            "theme_scores":  p.get("theme_scores", {}),
        })
        edges.append({"from": username, "to": pid, "label": "POSTED"})

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
                nodes.append({"id": thid, "label": theme.replace("_", " ").title(), "group": "theme", "size": 22})
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

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "username":        username,
            "total_posts":     len(posts),
            "total_hashtags":  sum(1 for k in seen if k.startswith("#")),
            "total_mentions":  sum(1 for k in seen if k.startswith("@")),
            "total_themes":    sum(1 for k in seen if k.startswith("theme:")),
            "filters_applied": filters or {},
        },
    }


def extract_comparison_subgraph(usernames: list[str], filters: Optional[dict] = None) -> dict[str, Any]:
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    seen: dict = {}

    for i, username in enumerate(usernames):
        posts = _posts(username)
        if filters:
            if "month"      in filters: posts = [p for p in posts if p["month"]      == filters["month"]]
            if "media_type" in filters: posts = [p for p in posts if p["media_type"] == filters["media_type"]]
            if "min_likes"  in filters: posts = [p for p in posts if p["like_count"] >= filters["min_likes"]]

        group      = f"brand_{chr(ord('a') + i)}"
        post_group = f"post_{chr(ord('a') + i)}"
        max_likes  = max((p["like_count"] for p in posts), default=1)
        brand_id   = f"brand:{username}"

        if brand_id not in seen:
            all_nodes.append({"id": brand_id, "label": f"@{username}", "group": group, "size": 50})
            seen[brand_id] = True

        for p in posts:
            pid  = f"{username}:{p['post_id']}"
            size = int(12 + 30 * (p["like_count"] / max(max_likes, 1)))
            all_nodes.append({
                "id":            pid,
                "label":         f"{username[:6]}\n{p['post_id'][:6]}",
                "group":         post_group,
                "size":          size,
                "like_count":    p["like_count"],
                "comment_count": p["comment_count"],
                "media_type":    p["media_type"],
                "url":           p["url"],
                "caption":       (p.get("caption") or "")[:120],
                "theme_scores":  p.get("theme_scores", {}),
            })
            all_edges.append({"from": brand_id, "to": pid, "label": "POSTED"})

            for tag in p.get("hashtags", []):
                tid = f"#{tag}"
                if tid not in seen:
                    all_nodes.append({"id": tid, "label": tid, "group": "hashtag", "size": 14})
                    seen[tid] = True
                all_edges.append({"from": pid, "to": tid, "label": "HAS_HASHTAG"})

            for mention in p.get("mentions", []):
                mid = f"@{mention}"
                if mid not in seen:
                    all_nodes.append({"id": mid, "label": mid, "group": "mention", "size": 16})
                    seen[mid] = True
                all_edges.append({"from": pid, "to": mid, "label": "MENTIONS"})

            for theme in p.get("themes", []):
                thid = f"theme:{theme}"
                if thid not in seen:
                    all_nodes.append({"id": thid, "label": theme.replace("_", " ").title(), "group": "theme", "size": 22})
                    seen[thid] = True
                all_edges.append({"from": pid, "to": thid, "label": "THEME"})

            month_id = f"month:{p['month']}"
            if month_id not in seen:
                all_nodes.append({"id": month_id, "label": p["month"], "group": "month", "size": 24})
                seen[month_id] = True
            all_edges.append({"from": pid, "to": month_id, "label": "BELONGS_TO"})

            mt_id = f"media:{p['media_type']}"
            if mt_id not in seen:
                all_nodes.append({"id": mt_id, "label": p["media_type"].upper(), "group": "media_type", "size": 20})
                seen[mt_id] = True
            all_edges.append({"from": pid, "to": mt_id, "label": "IS_TYPE"})

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "meta": {
            "usernames":       usernames,
            "total_nodes":     len(all_nodes),
            "total_edges":     len(all_edges),
            "filters_applied": filters or {},
        },
    }


def get_graph_context(username: str) -> dict[str, Any]:
    posts = _posts(username)
    if not posts:
        return {"error": f"No posts for @{username}."}
    return {
        "username":   username,
        "period":     f"last {DAYS_BACK} days",
        "engagement": get_engagement_summary(username),
        "hashtags":   get_hashtag_analysis(username),
        "monthly":    get_monthly_breakdown(username),
        "posts": [
            {
                "post_id":       p["post_id"],
                "url":           p["url"],
                "caption":       p["caption"],
                "hashtags":      p["hashtags"],
                "themes":        p.get("themes", []),
                "theme_scores":  p.get("theme_scores", {}),
                "like_count":    p["like_count"],
                "comment_count": p["comment_count"],
                "media_type":    p["media_type"],
                "month":         p["month"],
                "timestamp":     p["timestamp"],
            }
            for p in posts
        ],
    }


def get_comparison_context(usernames: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparison_period": f"last {DAYS_BACK} days",
        "accounts":          {},
        "shared_hashtags":   [],
        "shared_themes":     [],
    }
    hashtag_sets: list[set] = []
    theme_sets:   list[set] = []

    for username in usernames:
        try:
            ctx = get_graph_context(username)
            result["accounts"][username] = ctx
            tags   = {r["hashtag"].lstrip("#") for r in ctx["hashtags"].get("top_by_frequency", [])}
            themes = {t for p in ctx["posts"] for t in p.get("themes", [])}
            hashtag_sets.append(tags)
            theme_sets.append(themes)
        except Exception as e:
            result["accounts"][username] = {"error": str(e)}

    if len(hashtag_sets) >= 2:
        shared = hashtag_sets[0]
        for s in hashtag_sets[1:]:
            shared &= s
        result["shared_hashtags"] = sorted(shared)

    if len(theme_sets) >= 2:
        shared_t = theme_sets[0]
        for s in theme_sets[1:]:
            shared_t &= s
        result["shared_themes"] = sorted(shared_t)

    return result