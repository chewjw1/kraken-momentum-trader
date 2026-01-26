"""
Flask dashboard for scalping strategy monitoring.

Run with: python -m src.scalping_dashboard.app
Or: python -m src.scalping_dashboard.app --port 5001
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, jsonify


def create_app(data_dir: str = "data/scalping") -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'scalping-dashboard-secret-key')
    app.config['DATA_DIR'] = Path(data_dir)

    @app.route('/')
    def dashboard():
        """Main dashboard page."""
        state = load_state(app.config['DATA_DIR'])
        return render_template('scalping_dashboard.html', **state)

    @app.route('/api/status')
    def api_status():
        """JSON API for current status."""
        state = load_state(app.config['DATA_DIR'])
        return jsonify(state)

    @app.route('/api/pairs')
    def api_pairs():
        """JSON API for pair status."""
        state = load_state(app.config['DATA_DIR'])
        return jsonify(state.get('pairs', {}))

    return app


def load_state(data_dir: Path) -> dict:
    """Load state from JSON file."""
    state_file = data_dir / "state.json"

    default_state = {
        'capital': 10000.0,
        'positions': {},
        'metrics': {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'start_time': datetime.now(timezone.utc).isoformat()
        },
        'pairs': {},
        'enabled_pairs': [],
        'disabled_pairs': [],
        'last_update': 'Never',
        'paper_trading': True,
        'strategy': 'scalping'
    }

    if not state_file.exists():
        return default_state

    try:
        with open(state_file) as f:
            state = json.load(f)

        # Calculate win rate
        total = state.get('metrics', {}).get('total_trades', 0)
        wins = state.get('metrics', {}).get('wins', 0)
        if total > 0:
            state['metrics']['win_rate'] = (wins / total) * 100
        else:
            state['metrics']['win_rate'] = 0.0

        # Get enabled/disabled pairs from pair_manager
        pair_manager = state.get('pair_manager', {})
        pairs_data = pair_manager.get('pairs', {})

        enabled = []
        disabled = []
        pairs_status = {}

        for pair, data in pairs_data.items():
            pairs_status[pair] = {
                'enabled': data.get('is_enabled', True),
                'total_trades': data.get('total_trades', 0),
                'total_pnl': data.get('total_pnl', 0.0),
                'wins': data.get('wins', 0),
                'losses': data.get('losses', 0),
                'consecutive_losses': data.get('consecutive_losses', 0),
                'disabled_reason': data.get('disabled_reason'),
                'cooldown_until': data.get('cooldown_until')
            }
            if data.get('is_enabled', True):
                enabled.append(pair)
            else:
                disabled.append(pair)

        state['pairs'] = pairs_status
        state['enabled_pairs'] = enabled
        state['disabled_pairs'] = disabled

        return state

    except Exception as e:
        print(f"Error loading state: {e}")
        return default_state


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scalping Dashboard")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--data-dir", default="data/scalping", help="Data directory")
    args = parser.parse_args()

    app = create_app(args.data_dir)

    print(f"""
================================================================
           SCALPING DASHBOARD
================================================================
  URL: http://{args.host}:{args.port}
  Data: {args.data_dir}
================================================================
    """)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
