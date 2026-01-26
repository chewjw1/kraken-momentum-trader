"""
Run the momentum dashboard.

Usage: python -m src.dashboard --port 44490
"""

import argparse
from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="Momentum Trading Dashboard")
    parser.add_argument("--port", type=int, default=44490, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    app = create_app()

    print(f"""
================================================================
           MOMENTUM TRADER DASHBOARD
================================================================
  URL: http://{args.host}:{args.port}
================================================================
    """)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
