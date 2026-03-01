"""
Flask routes for the dashboard.
"""

from flask import Blueprint, Flask, jsonify, redirect, render_template, request, session, url_for

from .auth import check_auth, login_required
from .data_service import DashboardDataService

bp = Blueprint('main', __name__)
data_service: DashboardDataService = None


def init_app(app: Flask) -> None:
    """Initialize routes with the Flask app."""
    global data_service
    data_service = DashboardDataService(app.config.get('DB_PATH', 'data/trading.db'))
    app.register_blueprint(bp)


@bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    error = None
    if request.method == 'POST':
        if check_auth(request.form['username'], request.form['password']):
            session['logged_in'] = True
            next_url = request.args.get('next', url_for('main.dashboard'))
            return redirect(next_url)
        error = 'Invalid credentials'
    return render_template('login.html', error=error)


@bp.route('/logout')
def logout():
    """Logout and redirect to login."""
    session.pop('logged_in', None)
    return redirect(url_for('main.login'))


@bp.route('/')
@login_required
def dashboard():
    """Main dashboard page."""
    state = data_service.get_current_state()
    metrics = data_service.get_aggregate_metrics()
    recent_trades = data_service.get_recent_trades(10)
    pairs_summary = data_service.get_pairs_summary()

    # Extract positions from state
    positions = []
    state_machine = state.get('state_machine', {})
    if state_machine.get('positions'):
        for pair, pos in state_machine['positions'].items():
            entries = pos.get('entries', [])
            positions.append({
                'pair': pair,
                'side': pos.get('side', 'long'),
                'entry_price': entries[0].get('price', 0) if entries else 0,
                'size': sum(e.get('size', 0) for e in entries),
                'num_entries': len(entries),
                'total_cost_usd': sum(e.get('size_usd', 0) for e in entries)
            })

    # Get capital from state
    capital = state.get('metrics', {}).get('current_capital', 10000)

    # Get current state
    current_state = state_machine.get('state', 'unknown')

    # Get regime data
    regime_info = data_service.get_regime_info()
    global_regime = regime_info.get('_global', {})

    return render_template(
        'dashboard.html',
        metrics=metrics,
        positions=positions,
        recent_trades=recent_trades,
        pairs_summary=pairs_summary,
        capital=capital,
        state=current_state,
        global_regime=global_regime,
    )


@bp.route('/trades')
@login_required
def trades():
    """Trade history page with filtering."""
    page = request.args.get('page', 1, type=int)
    per_page = 25
    pair = request.args.get('pair')
    start_date = request.args.get('start')
    end_date = request.args.get('end')

    trades_list, total = data_service.get_trades(
        pair=pair,
        start_date=start_date,
        end_date=end_date,
        limit=per_page,
        offset=(page - 1) * per_page
    )

    # Get unique pairs for filter dropdown
    pairs_summary = data_service.get_pairs_summary()
    pairs = [p['pair'] for p in pairs_summary]

    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    return render_template(
        'trades.html',
        trades=trades_list,
        page=page,
        total_pages=total_pages,
        total=total,
        pairs=pairs,
        current_pair=pair,
        start_date=start_date,
        end_date=end_date
    )


@bp.route('/performance')
@login_required
def performance():
    """Performance page with charts."""
    daily_metrics = data_service.get_daily_metrics(90)
    state = data_service.get_current_state()
    initial_capital = state.get('metrics', {}).get('initial_capital', 10000)
    equity_curve = data_service.get_equity_curve(initial_capital)
    ml_stats = data_service.get_ml_stats()
    metrics = data_service.get_aggregate_metrics()

    return render_template(
        'performance.html',
        daily_metrics=daily_metrics,
        equity_curve=equity_curve,
        ml_stats=ml_stats,
        metrics=metrics
    )


# API endpoints for AJAX
@bp.route('/api/metrics')
@login_required
def api_metrics():
    """JSON endpoint for metrics."""
    return jsonify(data_service.get_aggregate_metrics())


@bp.route('/api/positions')
@login_required
def api_positions():
    """JSON endpoint for current positions."""
    state = data_service.get_current_state()
    positions = []
    state_machine = state.get('state_machine', {})
    if state_machine.get('positions'):
        for pair, pos in state_machine['positions'].items():
            entries = pos.get('entries', [])
            positions.append({
                'pair': pair,
                'side': pos.get('side', 'long'),
                'entries': entries,
                'num_entries': len(entries)
            })
    return jsonify(positions)


@bp.route('/api/pair/<path:pair>')
@login_required
def api_pair_detail(pair):
    """JSON endpoint for detailed pair metrics and trade history."""
    detail = data_service.get_pair_detail(pair)
    return jsonify(detail)


@bp.route('/api/regime')
@login_required
def api_regime():
    """JSON endpoint for regime information."""
    return jsonify(data_service.get_regime_info())
