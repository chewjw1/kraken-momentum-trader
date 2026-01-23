"""
Simple authentication for the dashboard.
Credentials are loaded from environment variables.
"""

import functools
import os

from flask import redirect, request, session, url_for


def check_auth(username: str, password: str) -> bool:
    """Verify username and password against environment variables."""
    expected_user = os.environ.get('DASHBOARD_USERNAME', 'admin')
    expected_pass = os.environ.get('DASHBOARD_PASSWORD', 'changeme')
    return username == expected_user and password == expected_pass


def login_required(f):
    """Decorator to require authentication for a route."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('main.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function
