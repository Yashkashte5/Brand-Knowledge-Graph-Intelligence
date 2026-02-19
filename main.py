import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    if not os.environ.get("APIFY_API_TOKEN"):
        print("Error: APIFY_API_TOKEN not set in .env")
        sys.exit(1)

    import asyncio
    import server
    asyncio.run(server.main())


if __name__ == "__main__":
    main()