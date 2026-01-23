#!/usr/bin/env python3
"""
Flask dashboard entry point for Kraken Momentum Trader.

Run with: python run_dashboard.py

Environment variables:
    DASHBOARD_USERNAME - Login username (default: admin)
    DASHBOARD_PASSWORD - Login password (default: changeme)
    DASHBOARD_SECRET_KEY - Flask session secret key
    DASHBOARD_HOST - Host to bind to (default: 0.0.0.0)
    DASHBOARD_PORT - Port to run on (default: 5000)
    DB_PATH - Path to trading database (default: data/trading.db)
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.app import create_app

app = create_app()

if __name__ == '__main__':
    host = os.environ.get('DASHBOARD_HOST', '0.0.0.0')
    port = int(os.environ.get('DASHBOARD_PORT', 5000))
    debug = os.environ.get('DASHBOARD_DEBUG', 'false').lower() == 'true'

    print(f"Starting Kraken Trader Dashboard on http://{host}:{port}")
    print("Press Ctrl+C to stop")

    app.run(host=host, port=port, debug=debug)
