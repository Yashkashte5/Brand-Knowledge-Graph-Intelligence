import asyncio
import json
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from graph import (
    build_graph,
    get_graph_context,
    get_engagement_summary,
    get_comparison_context,
    find_similar_posts,
)
from ingest import scrape_account, processed_path, get_cached_accounts
from visualize import generate_visualization, generate_comparison_visualization

log = logging.getLogger(__name__)
app = Server("brand-instagram-graph")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="scrape_account",
            description="Scrape or refresh Instagram data for an account. Uses cache if available. Set force=true to re-scrape.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "force":    {"type": "boolean", "default": False},
                },
                "required": ["username"],
            },
        ),
        types.Tool(
            name="get_account_context",
            description="Get the full knowledge graph context for an Instagram account. Auto-scrapes if not cached.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username":     {"type": "string"},
                    "force_scrape": {"type": "boolean", "default": False},
                },
                "required": ["username"],
            },
        ),
        types.Tool(
            name="get_engagement_summary",
            description="Get engagement metrics for an account: avg/median likes, best post, media breakdown, monthly trend. Auto-scrapes if needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username":     {"type": "string"},
                    "force_scrape": {"type": "boolean", "default": False},
                },
                "required": ["username"],
            },
        ),
        types.Tool(
            name="find_similar_posts",
            description="Find posts similar to a given post_id within one account using embeddings or hashtag overlap.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "post_id":  {"type": "string"},
                    "top_k":    {"type": "integer", "default": 5},
                },
                "required": ["username", "post_id"],
            },
        ),
        types.Tool(
            name="compare_accounts",
            description="Compare two or more Instagram accounts. Returns data for each + shared hashtags and themes. Auto-scrapes as needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "usernames":    {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "force_scrape": {"type": "boolean", "default": False},
                },
                "required": ["usernames"],
            },
        ),
        types.Tool(
            name="export_graph_visual",
            description="Generate and open an interactive HTML graph for one account. Optional filters: month (YYYY-MM), media_type, min_likes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "username":    {"type": "string"},
                    "month":       {"type": "string"},
                    "media_type":  {"type": "string", "enum": ["image", "video", "carousel"]},
                    "min_likes":   {"type": "integer"},
                    "force_scrape":{"type": "boolean", "default": False},
                },
                "required": ["username"],
            },
        ),
        types.Tool(
            name="export_comparison_visual",
            description="Generate and open an interactive HTML comparison graph for two or more accounts. Auto-scrapes as needed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "usernames":   {"type": "array", "items": {"type": "string"}, "minItems": 2},
                    "month":       {"type": "string"},
                    "media_type":  {"type": "string", "enum": ["image", "video", "carousel"]},
                    "min_likes":   {"type": "integer"},
                    "force_scrape":{"type": "boolean", "default": False},
                },
                "required": ["usernames"],
            },
        ),
        types.Tool(
            name="list_cached_accounts",
            description="List all Instagram accounts that have been scraped and cached locally.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    try:
        if   name == "scrape_account":           result = _scrape(arguments)
        elif name == "get_account_context":      result = _context(arguments)
        elif name == "get_engagement_summary":   result = _engagement(arguments)
        elif name == "find_similar_posts":       result = _similar(arguments)
        elif name == "compare_accounts":         result = _compare(arguments)
        elif name == "export_graph_visual":      result = _visual(arguments)
        elif name == "export_comparison_visual": result = _comparison_visual(arguments)
        elif name == "list_cached_accounts":     result = _list_cached()
        else:                                    result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        log.exception(f"Error in {name}")
        result = {"error": str(e)}
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


def _ensure_scraped(username: str, force: bool = False) -> str:
    username = username.lower().lstrip("@")
    pp = processed_path(username)
    if force or not pp.exists():
        scrape_account(username, force=force)
    else:
        try:
            if not json.loads(pp.read_text()):
                scrape_account(username, force=True)
        except Exception:
            scrape_account(username, force=True)
    return username


def _scrape(args: dict) -> dict:
    username = args.get("username", "").strip().lower().lstrip("@")
    if not username:
        return {"error": "username required"}
    posts = scrape_account(username, force=bool(args.get("force", False)))
    return {"status": "success", "username": username, "posts_scraped": len(posts)}


def _context(args: dict) -> dict:
    username = args.get("username", "").strip()
    if not username:
        return {"error": "username required"}
    username = _ensure_scraped(username, force=bool(args.get("force_scrape", False)))
    build_graph(username)
    return get_graph_context(username)


def _engagement(args: dict) -> dict:
    username = args.get("username", "").strip()
    if not username:
        return {"error": "username required"}
    username = _ensure_scraped(username, force=bool(args.get("force_scrape", False)))
    build_graph(username)
    return get_engagement_summary(username)


def _similar(args: dict) -> dict:
    username = args.get("username", "").strip()
    post_id  = args.get("post_id", "").strip()
    if not username or not post_id:
        return {"error": "username and post_id required"}
    username = _ensure_scraped(username)
    return find_similar_posts(username, post_id, top_k=int(args.get("top_k", 5)))


def _compare(args: dict) -> dict:
    usernames = [u.strip().lower().lstrip("@") for u in args.get("usernames", [])]
    if len(usernames) < 2:
        return {"error": "At least 2 usernames required"}
    force = bool(args.get("force_scrape", False))
    for u in usernames:
        _ensure_scraped(u, force=force)
        build_graph(u)
    return get_comparison_context(usernames)


def _visual(args: dict) -> dict:
    username = args.get("username", "").strip()
    if not username:
        return {"error": "username required"}
    username = _ensure_scraped(username, force=bool(args.get("force_scrape", False)))
    filters = {k: (int(v) if k == "min_likes" else v) for k, v in {
        "month":      args.get("month"),
        "media_type": args.get("media_type"),
        "min_likes":  args.get("min_likes"),
    }.items() if v}
    path     = generate_visualization(username, filters=filters or None)
    abs_path = str(path.resolve())
    file_url = "file:///" + abs_path.replace("\\", "/")
    webbrowser.open(file_url)
    return {"status": "success", "username": username, "file": abs_path, "url": file_url, "filters_applied": filters}


def _comparison_visual(args: dict) -> dict:
    usernames = [u.strip().lower().lstrip("@") for u in args.get("usernames", [])]
    if len(usernames) < 2:
        return {"error": "At least 2 usernames required"}
    force = bool(args.get("force_scrape", False))
    for u in usernames:
        _ensure_scraped(u, force=force)
    filters = {k: (int(v) if k == "min_likes" else v) for k, v in {
        "month":      args.get("month"),
        "media_type": args.get("media_type"),
        "min_likes":  args.get("min_likes"),
    }.items() if v}
    path     = generate_comparison_visualization(usernames, filters=filters or None)
    abs_path = str(path.resolve())
    file_url = "file:///" + abs_path.replace("\\", "/")
    webbrowser.open(file_url)
    return {"status": "success", "usernames": usernames, "file": abs_path, "url": file_url, "filters_applied": filters}


def _list_cached() -> dict:
    accounts = get_cached_accounts()
    return {"cached_accounts": accounts, "count": len(accounts)}


async def main():
    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())