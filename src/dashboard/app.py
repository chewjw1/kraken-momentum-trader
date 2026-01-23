"""
Flask application factory for the dashboard.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask


def create_app(config: dict = None) -> Flask:
    """Create and configure the Flask application."""
    load_dotenv()

    # Get the dashboard directory for templates
    dashboard_dir = Path(__file__).parent

    app = Flask(
        __name__,
        template_folder=str(dashboard_dir / 'templates')
    )

    # Configuration
    app.secret_key = os.environ.get('DASHBOARD_SECRET_KEY', 'dev-secret-change-me')
    app.config['DB_PATH'] = os.environ.get('DB_PATH', 'data/trading.db')

    if config:
        app.config.update(config)

    # Register routes
    from . import routes
    routes.init_app(app)

    return app
