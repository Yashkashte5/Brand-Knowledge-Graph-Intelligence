import json
import sys
from pathlib import Path


def ensure_data() -> bool:
    p = Path("data/nike_processed.json")
    if p.exists():
        try:
            content = p.read_text().strip()
            if content:
                posts = json.loads(content)
                if posts:
                    print(f"[main] Data ready — {len(posts)} posts")
                    return True
        except json.JSONDecodeError:
            print("[main] Processed file corrupted — re-ingesting …")

    print("[main] No data found — running ingestion …")
    from ingest import run_ingestion
    return len(run_ingestion()) > 0


def ensure_graph() -> bool:
    from graph import build_graph
    build_graph()
    return True


def main():
    print("\nNike Instagram Graph Intelligence")
    print("=" * 42)

    if not ensure_data():
        print("[main] Ingestion failed. Check your APIFY_API_TOKEN in .env")
        sys.exit(1)

    if not ensure_graph():
        print("[main] Graph build failed.")
        sys.exit(1)

    print("[main] System ready — start server.py to connect Claude Desktop\n")


if __name__ == "__main__":
    main()