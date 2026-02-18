import asyncio
import http.server
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


PROJECT_DIR = Path(__file__).parent.resolve()
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="[server] %(message)s")
log = logging.getLogger(__name__)

GRAPH_VIEW   = Path("graph_view.html")
STATIC_HOST  = "localhost"
STATIC_PORT  = int(os.getenv("NIKE_GRAPH_PORT", "8765"))

_http_started = False

def _start_file_server():
    global _http_started
    if _http_started:
        return

    class _H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass

    def _serve():
        os.chdir(str(Path(__file__).parent))
        with http.server.HTTPServer((STATIC_HOST, STATIC_PORT), _H) as s:
            s.serve_forever()

    threading.Thread(target=_serve, daemon=True).start()
    _http_started = True
    log.info(f"File server: http://{STATIC_HOST}:{STATIC_PORT}")


app = Server("nike-instagram-graph")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_graph_context",
            description=(
                "Returns the full Nike Instagram knowledge graph as structured data: "
                "all posts (captions, hashtags, likes, comments, media type, timestamp), "
                "hashtag frequency + engagement stats, monthly trends, and media type breakdown. "
                "Use this whenever the user asks ANYTHING about Nike's Instagram — "
                "hashtags, campaigns, engagement, themes, trends, best posts, etc. "
                "You (Claude) interpret the data and answer the user's question."
            ),
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="find_similar_posts",
            description="Find posts similar to a given post_id using caption embeddings or hashtag overlap.",
            inputSchema={
                "type": "object",
                "properties": {
                    "post_id": {"type": "string"},
                    "top_k":   {"type": "integer", "default": 5},
                },
                "required": ["post_id"],
            },
        ),
        types.Tool(
            name="get_engagement_summary",
            description="Return deterministic engagement metrics: avg/median likes, best post, media breakdown, monthly trend.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="export_graph_visual",
            description=(
                "Generate an interactive HTML knowledge graph and return a browser URL. "
                "Optional filters: month (YYYY-MM), media_type (image/video/carousel), min_likes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "month":      {"type": "string"},
                    "media_type": {"type": "string", "enum": ["image", "video", "carousel"]},
                    "min_likes":  {"type": "integer"},
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    log.info(f"Tool: {name}  args: {arguments}")
    try:
        if   name == "get_graph_context":      result = _context()
        elif name == "find_similar_posts":     result = _similar(arguments)
        elif name == "get_engagement_summary": result = _engagement()
        elif name == "export_graph_visual":    result = _visual(arguments)
        else:                                  result = {"error": f"Unknown tool: {name}"}
    except Exception as e:
        log.exception(f"Error in {name}")
        result = {"error": str(e)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


def _context() -> dict:
    from graph import get_graph_context
    return get_graph_context()


def _similar(args: dict) -> dict:
    from graph import find_similar_posts
    pid = args.get("post_id", "").strip()
    if not pid:
        return {"error": "post_id required"}
    return find_similar_posts(pid, top_k=int(args.get("top_k", 5)))


def _engagement() -> dict:
    from graph import get_engagement_summary
    return get_engagement_summary()


def _visual(args: dict) -> dict:
    from visualize import generate_visualization
    filters = {}
    if args.get("month"):      filters["month"]      = args["month"]
    if args.get("media_type"): filters["media_type"] = args["media_type"]
    if args.get("min_likes"):  filters["min_likes"]  = int(args["min_likes"])

    path = generate_visualization(filters=filters or None)
    _start_file_server()
    url = f"http://{STATIC_HOST}:{STATIC_PORT}/{path.name}"
    return {
        "url":             url,
        "file":            str(path.resolve()),
        "message":         f"Open {url} in your browser.",
        "filters_applied": filters,
    }


async def main():

    from graph import build_graph
    build_graph()

    async with stdio_server() as (r, w):
        await app.run(r, w, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())