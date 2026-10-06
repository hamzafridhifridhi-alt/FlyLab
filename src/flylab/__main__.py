"""Launch the FlyLab web interface.

Usage:
    uv run flylab            # http://127.0.0.1:8000
    uv run flylab --port 9000

Binds to localhost by default. The API can start, stop, and lesion the
simulation without authentication, so keep it local unless you add auth.
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyLab connectome simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(f"[flylab] warning: binding to {args.host} exposes unauthenticated "
              "simulation control to your network.")

    uvicorn.run(
        "flylab.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        ws_max_size=64 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
