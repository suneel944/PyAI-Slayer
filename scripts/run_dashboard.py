#!/usr/bin/env python3
"""Run the AI Testing Dashboard."""

import argparse
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from dashboard.api import run_dashboard  # noqa: E402


def main():
    """Run dashboard with CLI arguments."""
    parser = argparse.ArgumentParser(description="PyAI-Slayer Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to (default: 8080)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 PyAI-Slayer Dashboard")
    print("=" * 60)
    print(f"Starting dashboard on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    run_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
