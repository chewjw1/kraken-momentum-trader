"""
Persistence module for trading data storage.

Provides both file-based (JSON) and SQLite storage options.
SQLite is recommended for production use.
"""

from .file_store import FileStore, StateStore, TradeLogger
from .sqlite_store import SQLiteStore

__all__ = [
    "FileStore",
    "StateStore",
    "TradeLogger",
    "SQLiteStore",
]
